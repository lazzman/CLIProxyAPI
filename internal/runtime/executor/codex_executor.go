package executor

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sort"
	"strings"
	"time"

	codexauth "github.com/router-for-me/CLIProxyAPI/v7/internal/auth/codex"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/config"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/misc"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/thinking"
	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	cliproxyauth "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/auth"
	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	log "github.com/sirupsen/logrus"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
	"github.com/tiktoken-go/tokenizer"

	"github.com/gin-gonic/gin"
	"github.com/google/uuid"
)

const (
	codexUserAgent             = "codex-tui/0.118.0 (Ubuntu 24.4.0; x86_64) xterm-256color (codex-tui; 0.118.0)"
	codexOriginator            = "codex-tui"
	codexSandbox               = "seccomp"
	codexDefaultImageToolModel = "gpt-image-2"
	// Give non-stream /responses a short chance to reach EOF so keep-alive
	// connections can be reused without reintroducing long tail latency.
	codexCompletedDrainGracePeriod = 100 * time.Millisecond
)

var dataTag = []byte("data:")

// collectCodexOutputItemDone 收集 Codex output_item.done 的完整 item。
// Codex 流式响应偶尔在 completed 事件里给空 output，后续统一用这些 item 回填。
func collectCodexOutputItemDone(eventData []byte, outputItemsByIndex map[int64][]byte, outputItemsFallback *[][]byte) {
	itemResult := gjson.GetBytes(eventData, "item")
	if !itemResult.Exists() || itemResult.Type != gjson.JSON {
		return
	}
	outputIndexResult := gjson.GetBytes(eventData, "output_index")
	if outputIndexResult.Exists() {
		outputItemsByIndex[outputIndexResult.Int()] = []byte(itemResult.Raw)
		return
	}
	*outputItemsFallback = append(*outputItemsFallback, []byte(itemResult.Raw))
}

// patchCodexCompletedOutput 在 completed.output 为空时回填已完成 item。
// 一次性构造 JSON 数组可以避免逐个 sjson 追加造成重复拷贝。
func patchCodexCompletedOutput(eventData []byte, outputItemsByIndex map[int64][]byte, outputItemsFallback [][]byte) []byte {
	outputResult := gjson.GetBytes(eventData, "response.output")
	shouldPatchOutput := (!outputResult.Exists() || !outputResult.IsArray() || len(outputResult.Array()) == 0) &&
		(len(outputItemsByIndex) > 0 || len(outputItemsFallback) > 0)
	if !shouldPatchOutput {
		return eventData
	}

	items := orderedCodexOutputItems(outputItemsByIndex, outputItemsFallback)
	completedDataPatched, _ := sjson.SetRawBytes(eventData, "response.output", buildRawJSONArray(items))
	return completedDataPatched
}

// codexTerminalStreamContextLengthErr 将 Codex SSE 里的终态上下文长度错误转成稳定 statusErr。
func codexTerminalStreamContextLengthErr(eventData []byte) (statusErr, bool) {
	streamErr, body, ok := codexTerminalStreamErr(eventData)
	if !ok || !codexTerminalErrorIsContextLength(body) {
		return statusErr{}, false
	}
	return streamErr, true
}

// codexTerminalStreamErrorBody 从 Codex 终态 SSE 事件里提取统一 OpenAI 风格错误体。
func codexTerminalStreamErrorBody(eventData []byte) []byte {
	switch gjson.GetBytes(eventData, "type").String() {
	case "error":
		if body := codexTerminalErrorBody(eventData, "error"); len(body) > 0 {
			return body
		}
		return codexTerminalTopLevelErrorBody(eventData)
	case "response.failed":
		if body := codexTerminalErrorBody(eventData, "response.error"); len(body) > 0 {
			return body
		}
		return codexTerminalErrorBody(eventData, "error")
	default:
		return nil
	}
}

// codexTerminalErrorBody 将指定错误字段包装成下游已有错误处理可识别的结构。
func codexTerminalErrorBody(eventData []byte, path string) []byte {
	errorResult := gjson.GetBytes(eventData, path)
	if !errorResult.Exists() {
		return nil
	}
	body := []byte(`{"error":{}}`)
	if errorResult.Type == gjson.JSON {
		body, _ = sjson.SetRawBytes(body, "error", []byte(errorResult.Raw))
	} else if message := strings.TrimSpace(errorResult.String()); message != "" {
		body, _ = sjson.SetBytes(body, "error.message", message)
	}
	return codexTerminalErrorBodyWithMessageFallback(body, eventData)
}

// codexTerminalErrorBodyWithMessageFallback 在上游只给 code/type 时补齐 message，避免客户端收到空错误。
func codexTerminalErrorBodyWithMessageFallback(body []byte, eventData []byte) []byte {
	if strings.TrimSpace(gjson.GetBytes(body, "error.message").String()) != "" {
		return body
	}
	for _, path := range []string{"response.error.message", "response.error.code", "response.error.type"} {
		if message := strings.TrimSpace(gjson.GetBytes(eventData, path).String()); message != "" {
			body, _ = sjson.SetBytes(body, "error.message", message)
			return body
		}
	}
	for _, path := range []string{"error.code", "error.type"} {
		if message := strings.TrimSpace(gjson.GetBytes(body, path).String()); message != "" {
			body, _ = sjson.SetBytes(body, "error.message", message)
			return body
		}
	}
	return body
}

// codexTerminalTopLevelErrorBody 兼容 type=error 但错误字段位于事件顶层的 Codex SSE 形态。
func codexTerminalTopLevelErrorBody(eventData []byte) []byte {
	body := []byte(`{"error":{}}`)
	keys := map[string]string{"message": "message", "code": "code", "type": "error_type", "param": "param"}
	hasValue := false
	for target, source := range keys {
		value := strings.TrimSpace(gjson.GetBytes(eventData, source).String())
		if value == "" {
			continue
		}
		body, _ = sjson.SetBytes(body, "error."+target, value)
		hasValue = true
	}
	if !hasValue {
		return nil
	}
	return codexTerminalErrorBodyWithMessageFallback(body, eventData)
}

// codexTerminalErrorIsContextLength 判断错误体是否属于上下文长度超限，便于稳定映射为 400。
func codexTerminalErrorIsContextLength(body []byte) bool {
	errorCode := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.code").String()))
	message := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.message").String()))
	return errorCode == "context_length_exceeded" ||
		errorCode == "context_too_large" ||
		strings.Contains(message, "context window") ||
		strings.Contains(message, "context length") ||
		strings.Contains(message, "too many tokens")
}

// orderedCodexOutputItems 按 output_index 排序，并把缺少 index 的 item 追加到尾部。
func orderedCodexOutputItems(outputItemsByIndex map[int64][]byte, outputItemsFallback [][]byte) [][]byte {
	indexes := make([]int64, 0, len(outputItemsByIndex))
	for idx := range outputItemsByIndex {
		indexes = append(indexes, idx)
	}
	sort.Slice(indexes, func(i, j int) bool { return indexes[i] < indexes[j] })

	items := make([][]byte, 0, len(outputItemsByIndex)+len(outputItemsFallback))
	for _, idx := range indexes {
		items = append(items, outputItemsByIndex[idx])
	}
	return append(items, outputItemsFallback...)
}

// buildRawJSONArray 直接拼装已验证的 JSON item，避免循环调用 sjson 产生多次复制。
func buildRawJSONArray(items [][]byte) []byte {
	if len(items) == 0 {
		return []byte("[]")
	}

	var buf bytes.Buffer
	totalLen := 2
	for _, item := range items {
		totalLen += len(item)
	}
	if len(items) > 1 {
		totalLen += len(items) - 1
	}
	buf.Grow(totalLen)
	buf.WriteByte('[')
	for i, item := range items {
		if i > 0 {
			buf.WriteByte(',')
		}
		buf.Write(item)
	}
	buf.WriteByte(']')
	return buf.Bytes()
}

// CodexExecutor is a stateless executor for Codex (OpenAI Responses API entrypoint).
// If api_key is unavailable on auth, it falls back to legacy via ClientAdapter.
type CodexExecutor struct {
	cfg *config.Config
}

func NewCodexExecutor(cfg *config.Config) *CodexExecutor { return &CodexExecutor{cfg: cfg} }

func (e *CodexExecutor) Identifier() string { return "codex" }

// PrepareRequest injects Codex credentials into the outgoing HTTP request.
func (e *CodexExecutor) PrepareRequest(req *http.Request, auth *cliproxyauth.Auth) error {
	if req == nil {
		return nil
	}
	apiKey, _ := codexCreds(auth)
	if strings.TrimSpace(apiKey) != "" {
		req.Header.Set("Authorization", "Bearer "+apiKey)
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(req, attrs)
	return nil
}

// HttpRequest injects Codex credentials into the request and executes it.
func (e *CodexExecutor) HttpRequest(ctx context.Context, auth *cliproxyauth.Auth, req *http.Request) (*http.Response, error) {
	if req == nil {
		return nil, fmt.Errorf("codex executor: request is nil")
	}
	if ctx == nil {
		ctx = req.Context()
	}
	httpReq := req.WithContext(ctx)
	if err := e.PrepareRequest(httpReq, auth); err != nil {
		return nil, err
	}
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	return httpClient.Do(httpReq)
}

func (e *CodexExecutor) Execute(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	if opts.Alt == "responses/compact" {
		return e.executeCompact(ctx, auth, req, opts)
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	plan, err := e.prepareCodexRequestPlan(ctx, req, opts, codexPreparedRequestPlanExecute)
	if err != nil {
		return resp, err
	}
	body := plan.body
	originalPayload := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayload = opts.OriginalRequest
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	httpReq, err := newCodexHTTPRequest(ctx, url, body, plan.conversationID)
	if err != nil {
		return resp, err
	}
	applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = newCodexStatusErr(httpResp.StatusCode, b)
		err = wrapCodexReasoningReplayInvalidErr(plan.replayScope, httpResp.StatusCode, b, err)
		return resp, err
	}

	data, err := readCodexCompletedEvent(ctx, e.cfg, httpResp.Body, body, reporter, plan.replayScope)
	if err != nil {
		if _, ok := err.(statusErr); !ok {
			recordAPIResponseError(ctx, e.cfg, err)
		}
		return resp, err
	}

	var param any
	translateOriginalPayload, translateBody := originalPayload, body
	if !codexResponseTranslatorNeedsRequestPayloads(from) {
		translateOriginalPayload = nil
		translateBody = nil
	}
	out := sdktranslator.TranslateNonStream(ctx, sdktranslator.FromString("codex"), from, req.Model, translateOriginalPayload, translateBody, data, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out), Headers: httpResp.Header.Clone()}
	return resp, nil
}

func (e *CodexExecutor) executeCompact(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (resp cliproxyexecutor.Response, err error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	plan, err := e.prepareCodexRequestPlan(ctx, req, opts, codexPreparedRequestPlanCompact)
	if err != nil {
		return resp, err
	}
	body := plan.body
	originalPayload := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayload = opts.OriginalRequest
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses/compact"
	httpReq, err := newCodexHTTPRequest(ctx, url, body, plan.conversationID)
	if err != nil {
		return resp, err
	}
	applyCodexHeaders(httpReq, auth, apiKey, false, e.cfg)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})
	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	defer func() {
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
	}()
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		b, _ := io.ReadAll(httpResp.Body)
		appendAPIResponseChunk(ctx, e.cfg, b)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), b))
		err = newCodexStatusErr(httpResp.StatusCode, b)
		return resp, err
	}
	data, err := io.ReadAll(httpResp.Body)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return resp, err
	}
	appendAPIResponseChunk(ctx, e.cfg, data)
	reporter.publish(ctx, parseOpenAIUsage(data))
	reporter.ensurePublished(ctx)
	var param any
	translateOriginalPayload, translateBody := originalPayload, body
	if !codexResponseTranslatorNeedsRequestPayloads(from) {
		translateOriginalPayload = nil
		translateBody = nil
	}
	out := sdktranslator.TranslateNonStream(ctx, sdktranslator.FromString("openai-response"), from, req.Model, translateOriginalPayload, translateBody, data, &param)
	resp = cliproxyexecutor.Response{Payload: []byte(out), Headers: httpResp.Header.Clone()}
	return resp, nil
}

func (e *CodexExecutor) ExecuteStream(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (_ *cliproxyexecutor.StreamResult, err error) {
	if opts.Alt == "responses/compact" {
		return nil, statusErr{code: http.StatusBadRequest, msg: "streaming not supported for /responses/compact"}
	}
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	apiKey, baseURL := codexCreds(auth)
	if baseURL == "" {
		baseURL = "https://chatgpt.com/backend-api/codex"
	}

	reporter := newUsageReporter(ctx, e.Identifier(), baseModel, auth)
	defer reporter.trackFailure(ctx, &err)

	from := opts.SourceFormat
	plan, err := e.prepareCodexRequestPlan(ctx, req, opts, codexPreparedRequestPlanExecuteStream)
	if err != nil {
		return nil, err
	}
	body := plan.body
	originalPayload := req.Payload
	if len(opts.OriginalRequest) > 0 {
		originalPayload = opts.OriginalRequest
	}

	url := strings.TrimSuffix(baseURL, "/") + "/responses"
	httpReq, err := newCodexHTTPRequest(ctx, url, body, plan.conversationID)
	if err != nil {
		return nil, err
	}
	applyCodexHeaders(httpReq, auth, apiKey, true, e.cfg)
	var authID, authLabel, authType, authValue string
	if auth != nil {
		authID = auth.ID
		authLabel = auth.Label
		authType, authValue = auth.AccountInfo()
	}
	recordAPIRequest(ctx, e.cfg, upstreamRequestLog{
		URL:       url,
		Method:    http.MethodPost,
		Headers:   httpReq.Header.Clone(),
		Body:      body,
		Provider:  e.Identifier(),
		AuthID:    authID,
		AuthLabel: authLabel,
		AuthType:  authType,
		AuthValue: authValue,
	})

	httpClient := newProxyAwareHTTPClient(ctx, e.cfg, auth, 0)
	httpResp, err := httpClient.Do(httpReq)
	if err != nil {
		recordAPIResponseError(ctx, e.cfg, err)
		return nil, err
	}
	recordAPIResponseMetadata(ctx, e.cfg, httpResp.StatusCode, httpResp.Header.Clone())
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		data, readErr := io.ReadAll(httpResp.Body)
		if errClose := httpResp.Body.Close(); errClose != nil {
			log.Errorf("codex executor: close response body error: %v", errClose)
		}
		if readErr != nil {
			recordAPIResponseError(ctx, e.cfg, readErr)
			return nil, readErr
		}
		appendAPIResponseChunk(ctx, e.cfg, data)
		logWithRequestID(ctx).Debugf("request error, error status: %d, error message: %s", httpResp.StatusCode, summarizeErrorBody(httpResp.Header.Get("Content-Type"), data))
		err = newCodexStatusErr(httpResp.StatusCode, data)
		err = wrapCodexReasoningReplayInvalidErr(plan.replayScope, httpResp.StatusCode, data, err)
		return nil, err
	}
	out := make(chan cliproxyexecutor.StreamChunk)
	go func() {
		defer close(out)
		defer func() {
			if errClose := httpResp.Body.Close(); errClose != nil {
				log.Errorf("codex executor: close response body error: %v", errClose)
			}
		}()
		scanner := bufio.NewScanner(httpResp.Body)
		scanner.Buffer(nil, 52_428_800) // 50MB
		var param any
		outputItemsByIndex := make(map[int64][]byte)
		var outputItemsFallback [][]byte
		translateOriginalPayload, translateBody := originalPayload, body
		if !codexResponseTranslatorNeedsRequestPayloads(from) {
			translateOriginalPayload = nil
			translateBody = nil
		}
		for scanner.Scan() {
			line := scanner.Bytes()
			appendAPIResponseChunk(ctx, e.cfg, line)
			translatedLine := bytes.Clone(line)

			if bytes.HasPrefix(line, dataTag) {
				data := bytes.TrimSpace(line[5:])
				if streamErr, terminalBody, ok := codexTerminalStreamErr(data); ok {
					errOut := wrapCodexReasoningReplayInvalidErr(plan.replayScope, streamErr.StatusCode(), terminalBody, streamErr)
					recordAPIResponseError(ctx, e.cfg, errOut)
					reporter.publishFailure(ctx)
					if !sendStreamChunk(ctx, out, cliproxyexecutor.StreamChunk{Err: errOut}) {
						return
					}
					return
				}
				switch gjson.GetBytes(data, "type").String() {
				case "response.output_item.done":
					collectCodexOutputItemDone(data, outputItemsByIndex, &outputItemsFallback)
				case "response.completed":
					data = patchCodexCompletedOutput(data, outputItemsByIndex, outputItemsFallback)
					if detail, ok := parseCodexUsage(data); ok {
						reporter.publish(ctx, detail)
					}
					publishCodexImageToolUsage(ctx, reporter, body, data)
					reporter.ensurePublished(ctx)
					cacheCodexReasoningReplayFromCompleted(plan.replayScope, data)
					translatedLine = append([]byte("data: "), data...)
				}
			}

			chunks := sdktranslator.TranslateStream(ctx, sdktranslator.FromString("codex"), from, req.Model, translateOriginalPayload, translateBody, translatedLine, &param)
			for i := range chunks {
				if !sendStreamChunk(ctx, out, cliproxyexecutor.StreamChunk{Payload: []byte(chunks[i])}) {
					return
				}
			}
		}
		if errScan := scanner.Err(); errScan != nil {
			recordAPIResponseError(ctx, e.cfg, errScan)
			reporter.publishFailure(ctx)
			if !sendStreamChunk(ctx, out, cliproxyexecutor.StreamChunk{Err: errScan}) {
				return
			}
		}
	}()
	return &cliproxyexecutor.StreamResult{Headers: httpResp.Header.Clone(), Chunks: out}, nil
}

func (e *CodexExecutor) CountTokens(ctx context.Context, auth *cliproxyauth.Auth, req cliproxyexecutor.Request, opts cliproxyexecutor.Options) (cliproxyexecutor.Response, error) {
	baseModel := thinking.ParseSuffix(req.Model).ModelName

	from := opts.SourceFormat
	to := sdktranslator.FromString("codex")
	body := sdktranslator.TranslateRequest(from, to, baseModel, req.Payload, false)

	body, err := thinking.ApplyThinking(body, req.Model, from.String(), to.String(), e.Identifier())
	if err != nil {
		return cliproxyexecutor.Response{}, err
	}

	body, _ = sjson.SetBytes(body, "model", baseModel)
	body, _ = sjson.DeleteBytes(body, "previous_response_id")
	body, _ = sjson.DeleteBytes(body, "prompt_cache_retention")
	body, _ = sjson.DeleteBytes(body, "safety_identifier")
	body, _ = sjson.DeleteBytes(body, "stream_options")
	body, _ = sjson.SetBytes(body, "stream", false)
	body = normalizeCodexInstructions(body)

	enc, err := tokenizerForCodexModel(baseModel)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("codex executor: tokenizer init failed: %w", err)
	}

	count, err := countCodexInputTokens(enc, body)
	if err != nil {
		return cliproxyexecutor.Response{}, fmt.Errorf("codex executor: token counting failed: %w", err)
	}

	usageJSON := fmt.Sprintf(`{"response":{"usage":{"input_tokens":%d,"output_tokens":0,"total_tokens":%d}}}`, count, count)
	translated := sdktranslator.TranslateTokenCount(ctx, to, from, count, []byte(usageJSON))
	return cliproxyexecutor.Response{Payload: []byte(translated)}, nil
}

func normalizeCodexInstructions(body []byte) []byte {
	instructions := gjson.GetBytes(body, "instructions")
	if !instructions.Exists() || instructions.Type == gjson.Null {
		body, _ = sjson.SetBytes(body, "instructions", "")
	}
	return body
}

func tokenizerForCodexModel(model string) (tokenizer.Codec, error) {
	sanitized := strings.ToLower(strings.TrimSpace(model))
	switch {
	case sanitized == "":
		return tokenizer.Get(tokenizer.Cl100kBase)
	case strings.HasPrefix(sanitized, "gpt-5"):
		return tokenizer.ForModel(tokenizer.GPT5)
	case strings.HasPrefix(sanitized, "gpt-4.1"):
		return tokenizer.ForModel(tokenizer.GPT41)
	case strings.HasPrefix(sanitized, "gpt-4o"):
		return tokenizer.ForModel(tokenizer.GPT4o)
	case strings.HasPrefix(sanitized, "gpt-4"):
		return tokenizer.ForModel(tokenizer.GPT4)
	case strings.HasPrefix(sanitized, "gpt-3.5"), strings.HasPrefix(sanitized, "gpt-3"):
		return tokenizer.ForModel(tokenizer.GPT35Turbo)
	default:
		return tokenizer.Get(tokenizer.Cl100kBase)
	}
}

func countCodexInputTokens(enc tokenizer.Codec, body []byte) (int64, error) {
	if enc == nil {
		return 0, fmt.Errorf("encoder is nil")
	}
	if len(body) == 0 {
		return 0, nil
	}

	root := gjson.ParseBytes(body)
	var segments []string

	if inst := strings.TrimSpace(root.Get("instructions").String()); inst != "" {
		segments = append(segments, inst)
	}

	inputItems := root.Get("input")
	if inputItems.IsArray() {
		arr := inputItems.Array()
		for i := range arr {
			item := arr[i]
			switch item.Get("type").String() {
			case "message":
				content := item.Get("content")
				if content.IsArray() {
					parts := content.Array()
					for j := range parts {
						part := parts[j]
						if text := strings.TrimSpace(part.Get("text").String()); text != "" {
							segments = append(segments, text)
						}
					}
				}
			case "function_call":
				if name := strings.TrimSpace(item.Get("name").String()); name != "" {
					segments = append(segments, name)
				}
				if args := strings.TrimSpace(item.Get("arguments").String()); args != "" {
					segments = append(segments, args)
				}
			case "function_call_output":
				if out := strings.TrimSpace(item.Get("output").String()); out != "" {
					segments = append(segments, out)
				}
			default:
				if text := strings.TrimSpace(item.Get("text").String()); text != "" {
					segments = append(segments, text)
				}
			}
		}
	}

	tools := root.Get("tools")
	if tools.IsArray() {
		tarr := tools.Array()
		for i := range tarr {
			tool := tarr[i]
			if name := strings.TrimSpace(tool.Get("name").String()); name != "" {
				segments = append(segments, name)
			}
			if desc := strings.TrimSpace(tool.Get("description").String()); desc != "" {
				segments = append(segments, desc)
			}
			if params := tool.Get("parameters"); params.Exists() {
				val := params.Raw
				if params.Type == gjson.String {
					val = params.String()
				}
				if trimmed := strings.TrimSpace(val); trimmed != "" {
					segments = append(segments, trimmed)
				}
			}
		}
	}

	textFormat := root.Get("text.format")
	if textFormat.Exists() {
		if name := strings.TrimSpace(textFormat.Get("name").String()); name != "" {
			segments = append(segments, name)
		}
		if schema := textFormat.Get("schema"); schema.Exists() {
			val := schema.Raw
			if schema.Type == gjson.String {
				val = schema.String()
			}
			if trimmed := strings.TrimSpace(val); trimmed != "" {
				segments = append(segments, trimmed)
			}
		}
	}

	text := strings.Join(segments, "\n")
	if text == "" {
		return 0, nil
	}

	count, err := enc.Count(text)
	if err != nil {
		return 0, err
	}
	return int64(count), nil
}

func (e *CodexExecutor) Refresh(ctx context.Context, auth *cliproxyauth.Auth) (*cliproxyauth.Auth, error) {
	log.Debugf("codex executor: refresh called")
	if auth == nil {
		return nil, statusErr{code: 500, msg: "codex executor: auth is nil"}
	}
	var refreshToken string
	if auth.Metadata != nil {
		if v, ok := auth.Metadata["refresh_token"].(string); ok && v != "" {
			refreshToken = v
		}
	}
	if refreshToken == "" {
		return auth, nil
	}
	// refresh 必须优先复用 auth 自身 proxy_url，避免多出口场景下错走全局代理。
	svc := codexauth.NewCodexAuthWithProxyURL(e.cfg, auth.ProxyURL)
	td, err := svc.RefreshTokensWithRetry(ctx, refreshToken, 3)
	if err != nil {
		return nil, err
	}
	if auth.Metadata == nil {
		auth.Metadata = make(map[string]any)
	}
	auth.Metadata["id_token"] = td.IDToken
	auth.Metadata["access_token"] = td.AccessToken
	if td.RefreshToken != "" {
		auth.Metadata["refresh_token"] = td.RefreshToken
	}
	if td.AccountID != "" {
		auth.Metadata["account_id"] = td.AccountID
	}
	auth.Metadata["email"] = td.Email
	// Use unified key in files
	auth.Metadata["expired"] = td.Expire
	auth.Metadata["type"] = "codex"
	now := time.Now().Format(time.RFC3339)
	auth.Metadata["last_refresh"] = now
	return auth, nil
}

func (e *CodexExecutor) cacheHelper(ctx context.Context, from sdktranslator.Format, url string, req cliproxyexecutor.Request, rawJSON []byte) (*http.Request, error) {
	conversationID := codexPromptCacheID(ctx, from, req)
	if conversationID != "" {
		rawJSON = setJSONStringFieldIfNeeded(rawJSON, "prompt_cache_key", conversationID)
	}
	return newCodexHTTPRequest(ctx, url, rawJSON, conversationID)
}

func readCodexCompletedEvent(ctx context.Context, cfg *config.Config, body io.ReadCloser, requestBody []byte, reporter *usageReporter, replayScope codexReasoningReplayScope) ([]byte, error) {
	reader := bufio.NewReader(body)
	outputItemsByIndex := make(map[int64][]byte)
	var outputItemsFallback [][]byte
	for {
		line, err := reader.ReadBytes('\n')
		if len(line) > 0 {
			appendAPIResponseChunk(ctx, cfg, line)
			if bytes.HasPrefix(line, dataTag) {
				data := bytes.TrimSpace(line[len(dataTag):])
				if streamErr, terminalBody, ok := codexTerminalStreamErr(data); ok {
					return nil, wrapCodexReasoningReplayInvalidErr(replayScope, streamErr.StatusCode(), terminalBody, streamErr)
				}
				switch gjson.GetBytes(data, "type").String() {
				case "response.output_item.done":
					collectCodexOutputItemDone(data, outputItemsByIndex, &outputItemsFallback)
				case "response.completed":
					completedData := patchCodexCompletedOutput(data, outputItemsByIndex, outputItemsFallback)
					if detail, ok := parseCodexUsage(completedData); ok {
						reporter.publish(ctx, detail)
					}
					publishCodexImageToolUsage(ctx, reporter, requestBody, completedData)
					reporter.ensurePublished(ctx)
					cacheCodexReasoningReplayFromCompleted(replayScope, completedData)
					drainCodexCompletedBody(ctx, cfg, reader, body)
					return bytes.Clone(completedData), nil
				}
			}
		}
		if err == nil {
			continue
		}
		if err == io.EOF {
			break
		}
		return nil, err
	}
	return nil, statusErr{code: 408, msg: "stream error: stream disconnected before completion: stream closed before response.completed"}
}

func drainCodexCompletedBody(ctx context.Context, cfg *config.Config, reader *bufio.Reader, body io.Closer) {
	if reader == nil || body == nil {
		return
	}

	done := make(chan struct{})
	go func() {
		defer close(done)
		buf := make([]byte, 4096)
		for {
			n, err := reader.Read(buf)
			if n > 0 {
				appendAPIResponseChunk(ctx, cfg, buf[:n])
			}
			if err != nil {
				return
			}
		}
	}()

	timer := time.NewTimer(codexCompletedDrainGracePeriod)
	defer timer.Stop()

	select {
	case <-done:
		return
	case <-timer.C:
		_ = body.Close()
		<-done
	}
}

func applyCodexHeaders(r *http.Request, auth *cliproxyauth.Auth, token string, stream bool, cfg *config.Config) {
	r.Header.Set("Content-Type", "application/json")
	r.Header.Set("Authorization", "Bearer "+token)

	var ginHeaders http.Header
	if ginCtx, ok := r.Context().Value("gin").(*gin.Context); ok && ginCtx != nil && ginCtx.Request != nil {
		ginHeaders = ginCtx.Request.Header
	}
	sourceUserAgent := firstNonEmptyTrimmed(r.Header.Get("User-Agent"), headerValue(ginHeaders, "User-Agent"))

	misc.EnsureHeader(r.Header, ginHeaders, "Version", "")
	misc.EnsureHeader(r.Header, ginHeaders, "Session_id", uuid.NewString())
	cfgUserAgent, _ := codexHeaderDefaults(cfg, auth)
	ensureHeaderWithConfigPrecedence(r.Header, ginHeaders, "User-Agent", cfgUserAgent, codexUserAgent)
	misc.EnsureHeader(r.Header, ginHeaders, "X-Client-Request-Id", r.Header.Get("Session_id"))
	ensureCodexTurnMetadata(r.Header, ginHeaders)

	if stream {
		r.Header.Set("Accept", "text/event-stream")
	} else {
		r.Header.Set("Accept", "application/json")
	}
	r.Header.Set("Connection", "close")
	r.Close = true

	isAPIKey := false
	if auth != nil && auth.Attributes != nil {
		if v := strings.TrimSpace(auth.Attributes["api_key"]); v != "" {
			isAPIKey = true
		}
	}
	if originator := strings.TrimSpace(headerValue(ginHeaders, "Originator")); originator != "" {
		r.Header.Set("Originator", originator)
	}
	if !isAPIKey {
		if strings.TrimSpace(r.Header.Get("Originator")) == "" {
			r.Header.Set("Originator", codexOriginator)
		}
		if auth != nil && auth.Metadata != nil {
			if accountID, ok := auth.Metadata["account_id"].(string); ok {
				r.Header.Set("Chatgpt-Account-Id", accountID)
			}
		}
	}
	var attrs map[string]string
	if auth != nil {
		attrs = auth.Attributes
	}
	util.ApplyCustomHeadersFromAttrs(r, attrs)
	applyCodexAuthFileUserAgent(r.Header, auth, sourceUserAgent, codexDefaultUserAgent(cfg, auth))
}

func codexAuthFileUserAgent(auth *cliproxyauth.Auth) string {
	if auth == nil {
		return ""
	}
	if !strings.EqualFold(strings.TrimSpace(auth.Provider), "codex") {
		if auth.Metadata == nil {
			return ""
		}
		if rawType, ok := auth.Metadata["type"].(string); !ok || !strings.EqualFold(strings.TrimSpace(rawType), "codex") {
			return ""
		}
	}
	if auth.Metadata == nil {
		return ""
	}
	if raw, ok := auth.Metadata["cli_ua"].(string); ok {
		return strings.TrimSpace(raw)
	}
	return ""
}

func applyCodexAuthFileUserAgent(headers http.Header, auth *cliproxyauth.Auth, sourceUserAgent, defaultUserAgent string) {
	if headers == nil {
		return
	}
	if ua := codexAuthFileUserAgent(auth); ua != "" {
		headers.Set("User-Agent", ua)
		return
	}
	sourceUserAgent = strings.TrimSpace(sourceUserAgent)
	defaultUserAgent = strings.TrimSpace(defaultUserAgent)
	if sourceUserAgent != "" && strings.Contains(strings.ToLower(sourceUserAgent), "codex") {
		headers.Set("User-Agent", sourceUserAgent)
		if auth != nil {
			if auth.Metadata == nil {
				auth.Metadata = make(map[string]any)
			}
			auth.Metadata["cli_ua"] = sourceUserAgent
		}
		return
	}
	if defaultUserAgent != "" {
		headers.Set("User-Agent", defaultUserAgent)
	}
}

func codexDefaultUserAgent(cfg *config.Config, auth *cliproxyauth.Auth) string {
	cfgUserAgent, _ := codexHeaderDefaults(cfg, auth)
	return firstNonEmptyTrimmed(cfgUserAgent, codexUserAgent)
}

func firstNonEmptyTrimmed(values ...string) string {
	for _, value := range values {
		if trimmed := strings.TrimSpace(value); trimmed != "" {
			return trimmed
		}
	}
	return ""
}

func headerValue(headers http.Header, key string) string {
	if headers == nil {
		return ""
	}
	return headers.Get(key)
}

func ensureCodexTurnMetadata(target http.Header, source http.Header) {
	if target == nil {
		return
	}
	if source != nil {
		if val := strings.TrimSpace(source.Get("X-Codex-Turn-Metadata")); val != "" {
			target.Set("X-Codex-Turn-Metadata", val)
			return
		}
	}
	if strings.TrimSpace(target.Get("X-Codex-Turn-Metadata")) != "" {
		return
	}

	seed := strings.TrimSpace(target.Get("X-Client-Request-Id"))
	if seed == "" {
		seed = strings.TrimSpace(target.Get("Session_id"))
	}
	if seed == "" {
		seed = uuid.NewString()
	}

	sessionID := strings.TrimSpace(target.Get("Session_id"))
	if sessionID == "" {
		sessionID = uuid.NewSHA1(uuid.NameSpaceOID, []byte("cli-proxy-api:codex:session:"+seed)).String()
		target.Set("Session_id", sessionID)
	}

	turnID := uuid.NewSHA1(uuid.NameSpaceOID, []byte("cli-proxy-api:codex:turn:"+seed)).String()
	payload := map[string]any{
		"session_id": sessionID,
		"turn_id":    turnID,
		"sandbox":    codexSandbox,
	}
	if encoded, err := json.Marshal(payload); err == nil {
		target.Set("X-Codex-Turn-Metadata", string(encoded))
	}
}

func newCodexStatusErr(statusCode int, body []byte) statusErr {
	errCode := statusCode
	if isCodexModelCapacityError(body) || isCodexUsageLimitError(body) {
		errCode = http.StatusTooManyRequests
	}
	body = classifyCodexStatusError(errCode, body)
	err := statusErr{code: errCode, msg: string(body)}
	if retryAfter := parseCodexRetryAfter(errCode, body, time.Now()); retryAfter != nil {
		err.retryAfter = retryAfter
	}
	return err
}

func classifyCodexStatusError(statusCode int, body []byte) []byte {
	code, errType, ok := codexStatusErrorClassification(statusCode, body)
	if !ok {
		return body
	}
	message := gjson.GetBytes(body, "error.message").String()
	if message == "" {
		message = gjson.GetBytes(body, "message").String()
	}
	if message == "" {
		message = strings.TrimSpace(string(body))
	}
	if message == "" {
		message = http.StatusText(statusCode)
	}
	out := []byte(`{"error":{}}`)
	out, _ = sjson.SetBytes(out, "error.message", message)
	out, _ = sjson.SetBytes(out, "error.type", errType)
	out, _ = sjson.SetBytes(out, "error.code", code)
	return out
}

func codexStatusErrorClassification(statusCode int, body []byte) (code string, errType string, ok bool) {
	errorMessage := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.message").String()))
	if errorMessage == "" {
		errorMessage = strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "message").String()))
	}
	lower := strings.ToLower(strings.TrimSpace(string(body)))
	upstreamCode := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.code").String()))
	upstreamType := strings.ToLower(strings.TrimSpace(gjson.GetBytes(body, "error.type").String()))
	isInvalidRequest := upstreamType == "" || upstreamType == "invalid_request_error"

	switch {
	case statusCode == http.StatusRequestEntityTooLarge ||
		upstreamCode == "context_length_exceeded" ||
		upstreamCode == "context_too_large" ||
		isInvalidRequest && (strings.Contains(errorMessage, "context length") ||
			strings.Contains(errorMessage, "context_length") ||
			strings.Contains(errorMessage, "maximum context") ||
			strings.Contains(errorMessage, "too many tokens")):
		return "context_too_large", "invalid_request_error", true
	case strings.Contains(lower, "invalid signature in thinking block") ||
		strings.Contains(lower, "invalid_encrypted_content"):
		return "thinking_signature_invalid", "invalid_request_error", true
	case upstreamCode == "previous_response_not_found" ||
		strings.Contains(lower, "previous_response_not_found") ||
		(strings.Contains(lower, "previous_response_id") && strings.Contains(lower, "not found")):
		return "previous_response_not_found", "invalid_request_error", true
	case statusCode == http.StatusUnauthorized ||
		upstreamType == "authentication_error" ||
		upstreamCode == "invalid_api_key" ||
		strings.Contains(lower, "invalid or expired token") ||
		strings.Contains(lower, "refresh_token_reused"):
		return "auth_unavailable", "authentication_error", true
	default:
		return "", "", false
	}
}

func publishCodexImageToolUsage(ctx context.Context, reporter *usageReporter, requestBody []byte, completedData []byte) {
	if !cliproxyexecutor.RequestHasExplicitImageGenerationIntent(requestBody) {
		return
	}
	if !codexResponseUsedImageGenerationTool(completedData) {
		return
	}
	detail, _ := parseCodexImageToolUsage(completedData)
	reporter.ensurePublished(ctx)
	reporter.publishAdditionalModel(ctx, codexImageGenerationToolModel(requestBody), detail)
}

func codexImageGenerationToolModel(body []byte) string {
	tools := gjson.GetBytes(body, "tools")
	if tools.IsArray() {
		for _, tool := range tools.Array() {
			if tool.Get("type").String() != "image_generation" {
				continue
			}
			if model := strings.TrimSpace(tool.Get("model").String()); model != "" {
				return model
			}
			break
		}
	}
	return codexDefaultImageToolModel
}

func isCodexModelCapacityError(errorBody []byte) bool {
	if len(errorBody) == 0 {
		return false
	}
	candidates := []string{
		gjson.GetBytes(errorBody, "error.message").String(),
		gjson.GetBytes(errorBody, "message").String(),
		string(errorBody),
	}
	for _, candidate := range candidates {
		lower := strings.ToLower(strings.TrimSpace(candidate))
		if lower == "" {
			continue
		}
		if strings.Contains(lower, "selected model is at capacity") ||
			strings.Contains(lower, "model is at capacity. please try a different model") {
			return true
		}
	}
	return false
}

func isCodexUsageLimitError(errorBody []byte) bool {
	return strings.EqualFold(strings.TrimSpace(gjson.GetBytes(errorBody, "error.type").String()), "usage_limit_reached") ||
		strings.EqualFold(strings.TrimSpace(gjson.GetBytes(errorBody, "type").String()), "usage_limit_reached")
}

func parseCodexRetryAfter(statusCode int, errorBody []byte, now time.Time) *time.Duration {
	if statusCode != http.StatusTooManyRequests || len(errorBody) == 0 {
		return nil
	}
	if strings.TrimSpace(gjson.GetBytes(errorBody, "error.type").String()) != "usage_limit_reached" {
		return nil
	}
	if resetsAt := gjson.GetBytes(errorBody, "error.resets_at").Int(); resetsAt > 0 {
		resetAtTime := time.Unix(resetsAt, 0)
		if resetAtTime.After(now) {
			retryAfter := resetAtTime.Sub(now)
			return &retryAfter
		}
	}
	if resetsInSeconds := gjson.GetBytes(errorBody, "error.resets_in_seconds").Int(); resetsInSeconds > 0 {
		retryAfter := time.Duration(resetsInSeconds) * time.Second
		return &retryAfter
	}
	return nil
}

func codexCreds(a *cliproxyauth.Auth) (apiKey, baseURL string) {
	if a == nil {
		return "", ""
	}
	if a.Attributes != nil {
		apiKey = a.Attributes["api_key"]
		baseURL = a.Attributes["base_url"]
	}
	if apiKey == "" && a.Metadata != nil {
		if v, ok := a.Metadata["access_token"].(string); ok {
			apiKey = v
		}
	}
	return
}

func (e *CodexExecutor) resolveCodexConfig(auth *cliproxyauth.Auth) *config.CodexKey {
	if auth == nil || e.cfg == nil {
		return nil
	}
	var attrKey, attrBase string
	if auth.Attributes != nil {
		attrKey = strings.TrimSpace(auth.Attributes["api_key"])
		attrBase = strings.TrimSpace(auth.Attributes["base_url"])
	}
	for i := range e.cfg.CodexKey {
		entry := &e.cfg.CodexKey[i]
		cfgKey := strings.TrimSpace(entry.APIKey)
		cfgBase := strings.TrimSpace(entry.BaseURL)
		if attrKey != "" && attrBase != "" {
			if strings.EqualFold(cfgKey, attrKey) && strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
			continue
		}
		if attrKey != "" && strings.EqualFold(cfgKey, attrKey) {
			if cfgBase == "" || strings.EqualFold(cfgBase, attrBase) {
				return entry
			}
		}
		if attrKey == "" && attrBase != "" && strings.EqualFold(cfgBase, attrBase) {
			return entry
		}
	}
	if attrKey != "" {
		for i := range e.cfg.CodexKey {
			entry := &e.cfg.CodexKey[i]
			if strings.EqualFold(strings.TrimSpace(entry.APIKey), attrKey) {
				return entry
			}
		}
	}
	return nil
}
