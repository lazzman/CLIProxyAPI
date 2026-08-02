// Package claude provides response translation functionality for Codex to Claude Code API compatibility.
// This package handles the conversion of Codex API responses into Claude Code-compatible
// Server-Sent Events (SSE) format, implementing a sophisticated state machine that manages
// different response types including text content, thinking processes, and function calls.
// The translation ensures proper sequencing of SSE events and maintains state across
// multiple response chunks to provide a seamless streaming experience.
package claude

import (
	"bytes"
	"context"
	"fmt"
	"strings"

	"github.com/router-for-me/CLIProxyAPI/v7/internal/util"
	"github.com/tidwall/gjson"
	"github.com/tidwall/sjson"
)

var (
	dataTag = []byte("data:")
)

const codexThinkingSummaryPartSeparator = "\n\n"

// ConvertCodexResponseToClaudeParams holds parameters for response conversion.
type ConvertCodexResponseToClaudeParams struct {
	HasCompletedToolCall bool
	BlockIndex           int
	HasTextDelta         bool
	TextBlockOpen        bool
	ThinkingBlockOpen    bool
	ThinkingSignature    string
	ThinkingSummarySeen  bool
	FunctionCalls        map[string]*codexFunctionCallStream
	FunctionCallQueue    []*codexFunctionCallStream
	ActiveFunctionCall   *codexFunctionCallStream
	LastFunctionCall     *codexFunctionCallStream
	DeferredStreamEvents [][]byte
}

type codexFunctionCallStream struct {
	CallID                    string
	Name                      string
	BlockIndex                int
	Arguments                 string
	EmittedArgumentsLength    int
	HasReceivedArgumentsDelta bool
	EmitInitialEmptyDelta     bool
	Started                   bool
	Done                      bool
	Closed                    bool
}

func appendClaudeSSEEvent(output *strings.Builder, event, payload string) {
	output.WriteString("event: ")
	output.WriteString(event)
	output.WriteByte('\n')
	output.WriteString("data: ")
	output.WriteString(payload)
	output.WriteString("\n\n")
}

// ConvertCodexResponseToClaude performs sophisticated streaming response format conversion.
// This function implements a complex state machine that translates Codex API responses
// into Claude Code-compatible Server-Sent Events (SSE) format. It manages different response types
// and handles state transitions between content blocks, thinking processes, and function calls.
//
// Response type states: 0=none, 1=content, 2=thinking, 3=function
// The function maintains state across multiple calls to ensure proper SSE event sequencing.
//
// Parameters:
//   - ctx: The context for the request, used for cancellation and timeout handling
//   - modelName: The name of the model being used for the response (unused in current implementation)
//   - rawJSON: The raw JSON response from the Codex API
//   - param: A pointer to a parameter object for maintaining state between calls
//
// Returns:
//   - []string: A slice of strings, each containing a Claude Code-compatible JSON response
func ConvertCodexResponseToClaude(_ context.Context, _ string, originalRequestRawJSON, _ []byte, rawJSON []byte, param *any) []string {
	if *param == nil {
		*param = &ConvertCodexResponseToClaudeParams{
			BlockIndex: 0,
		}
	}

	if !bytes.HasPrefix(rawJSON, dataTag) {
		return []string{}
	}
	streamEventRawJSON := bytes.Clone(rawJSON)
	rawJSON = bytes.TrimSpace(rawJSON[5:])

	params := (*param).(*ConvertCodexResponseToClaudeParams)
	rootResult := gjson.ParseBytes(rawJSON)
	typeStr := rootResult.Get("type").String()
	if params.ActiveFunctionCall != nil && shouldDeferCodexStreamEvent(typeStr, rootResult) {
		params.DeferredStreamEvents = append(params.DeferredStreamEvents, streamEventRawJSON)
		return []string{}
	}
	var output strings.Builder
	template := ""

	switch typeStr {
	case "error":
		appendClaudeStreamError(&output, rootResult)

	case "response.created":
		template = `{"type":"message_start","message":{"id":"","type":"message","role":"assistant","model":"claude-opus-4-1-20250805","stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0},"content":[],"stop_reason":null}}`
		template, _ = sjson.Set(template, "message.model", rootResult.Get("response.model").String())
		template, _ = sjson.Set(template, "message.id", rootResult.Get("response.id").String())
		appendClaudeSSEEvent(&output, "message_start", template)

	case "response.reasoning_summary_part.added":
		closeOpenTextBlock(params, &output)
		if params.ThinkingBlockOpen {
			appendCodexThinkingDelta(params, &output, codexThinkingSummaryPartSeparator)
		} else {
			startCodexThinkingBlock(params, &output)
		}
		params.ThinkingSummarySeen = true

	case "response.reasoning_summary_text.delta":
		closeOpenTextBlock(params, &output)
		startCodexThinkingBlock(params, &output)
		appendCodexThinkingDelta(params, &output, rootResult.Get("delta").String())

	case "response.reasoning_summary_part.done":
		// One Codex reasoning item may contain several summary parts. Its final
		// signature only arrives with output_item.done, so keep the block open.

	case "response.content_part.added":
		finalizeCodexThinkingBlock(params, &output)
		partType := rootResult.Get("part.type").String()
		if partType == "" || partType == "output_text" {
			startCodexTextBlock(params, &output)
		}

	case "response.output_text.delta":
		params.HasTextDelta = true
		finalizeCodexThinkingBlock(params, &output)
		startCodexTextBlock(params, &output)
		template = `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`
		template, _ = sjson.Set(template, "index", params.BlockIndex)
		template, _ = sjson.Set(template, "delta.text", rootResult.Get("delta").String())
		appendClaudeSSEEvent(&output, "content_block_delta", template)

	case "response.content_part.done":
		partType := rootResult.Get("part.type").String()
		if partType == "" || partType == "output_text" {
			closeOpenTextBlock(params, &output)
		}

	case "response.completed", "response.incomplete":
		template = `{"type":"message_delta","delta":{"stop_reason":"tool_use","stop_sequence":null},"usage":{"input_tokens":0,"output_tokens":0}}`
		responseData := rootResult.Get("response")
		finalizeCodexThinkingBlock(params, &output)
		closeOpenTextBlock(params, &output)
		completed := typeStr == "response.completed"
		appendCodexFunctionCallsFromTerminal(&output, params, originalRequestRawJSON, responseData, completed)
		appendDeferredCodexStreamEvents(&output, originalRequestRawJSON, param)
		finalizeCodexThinkingBlock(params, &output)
		closeOpenTextBlock(params, &output)
		hasCompletedToolCall := params.HasCompletedToolCall && completed
		template, _ = sjson.Set(template, "delta.stop_reason", mapCodexStopReasonToClaude(codexStopReason(responseData), hasCompletedToolCall))
		template = setClaudeStopSequence(template, "delta.stop_sequence", responseData)
		inputTokens, outputTokens, cachedTokens := extractResponsesUsage(responseData.Get("usage"))
		template, _ = sjson.Set(template, "usage.input_tokens", inputTokens)
		template, _ = sjson.Set(template, "usage.output_tokens", outputTokens)
		if cachedTokens > 0 {
			template, _ = sjson.Set(template, "usage.cache_read_input_tokens", cachedTokens)
		}
		appendClaudeSSEEvent(&output, "message_delta", template)
		appendClaudeSSEEvent(&output, "message_stop", `{"type":"message_stop"}`)

	case "response.output_item.added":
		itemResult := rootResult.Get("item")
		switch itemResult.Get("type").String() {
		case "function_call":
			finalizeCodexThinkingBlock(params, &output)
			closeOpenTextBlock(params, &output)
			call := recordCodexFunctionCall(params, rootResult, itemResult)
			updateCodexFunctionCallIdentity(params, call, rootResult, itemResult)
			call.EmitInitialEmptyDelta = call.Name != ""
			appendCodexFunctionCallQueue(&output, params, originalRequestRawJSON)

		case "reasoning":
			closeOpenTextBlock(params, &output)
			finalizeCodexThinkingBlock(params, &output)
			params.ThinkingSummarySeen = false
			params.ThinkingSignature = itemResult.Get("encrypted_content").String()
		}

	case "response.output_item.done":
		itemResult := rootResult.Get("item")
		switch itemResult.Get("type").String() {
		case "message":
			// 兜底：当 Codex 没有逐段下发 output_text.delta 时，
			// 仍要从最终 message 的 output_text 中补出 Claude 文本块。
			if params.HasTextDelta {
				break
			}
			contentResult := itemResult.Get("content")
			if !contentResult.Exists() || !contentResult.IsArray() {
				break
			}
			var textBuilder strings.Builder
			contentResult.ForEach(func(_, part gjson.Result) bool {
				if part.Get("type").String() != "output_text" {
					return true
				}
				if txt := part.Get("text").String(); txt != "" {
					textBuilder.WriteString(txt)
				}
				return true
			})
			text := textBuilder.String()
			if text == "" {
				break
			}

			finalizeCodexThinkingBlock(params, &output)
			startCodexTextBlock(params, &output)

			template = `{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":""}}`
			template, _ = sjson.Set(template, "index", params.BlockIndex)
			template, _ = sjson.Set(template, "delta.text", text)
			appendClaudeSSEEvent(&output, "content_block_delta", template)

			params.HasTextDelta = true
			closeOpenTextBlock(params, &output)
		case "function_call":
			finalizeCodexThinkingBlock(params, &output)
			closeOpenTextBlock(params, &output)
			call := codexFunctionCallForEvent(params, rootResult, itemResult)
			if call == nil {
				call = recordCodexFunctionCall(params, rootResult, itemResult)
			}
			updateCodexFunctionCallIdentity(params, call, rootResult, itemResult)
			updateCodexFunctionCallArguments(call, itemResult.Get("arguments").String(), false)
			call.Done = true
			params.HasCompletedToolCall = true
			appendCodexFunctionCallQueue(&output, params, originalRequestRawJSON)
		case "reasoning":
			closeOpenTextBlock(params, &output)
			if signature := itemResult.Get("encrypted_content").String(); signature != "" {
				params.ThinkingSignature = signature
			}
			if params.ThinkingSummarySeen {
				finalizeCodexThinkingBlock(params, &output)
			} else {
				finalizeCodexSignatureOnlyThinkingBlock(params, &output)
			}
			params.ThinkingSignature = ""
			params.ThinkingSummarySeen = false
		}

	case "response.function_call_arguments.delta":
		call := codexFunctionCallForEvent(params, rootResult, gjson.Result{})
		if call == nil {
			call = recordCodexFunctionCall(params, rootResult, gjson.Result{})
		}
		updateCodexFunctionCallArguments(call, rootResult.Get("delta").String(), true)
		appendCodexFunctionCallBufferedArguments(&output, params, call)

	case "response.function_call_arguments.done":
		call := codexFunctionCallForEvent(params, rootResult, gjson.Result{})
		if call == nil {
			call = recordCodexFunctionCall(params, rootResult, gjson.Result{})
		}
		updateCodexFunctionCallArguments(call, rootResult.Get("arguments").String(), false)
		appendCodexFunctionCallBufferedArguments(&output, params, call)
	}

	if len(params.FunctionCallQueue) == 0 {
		appendDeferredCodexStreamEvents(&output, originalRequestRawJSON, param)
	}
	return []string{output.String()}
}

func shouldDeferCodexStreamEvent(typeStr string, rootResult gjson.Result) bool {
	switch typeStr {
	case "error", "response.completed", "response.incomplete", "response.function_call_arguments.delta", "response.function_call_arguments.done":
		return false
	case "response.output_item.added", "response.output_item.done":
		return rootResult.Get("item.type").String() != "function_call"
	default:
		return true
	}
}

func appendDeferredCodexStreamEvents(output *strings.Builder, originalRequestRawJSON []byte, param *any) {
	if output == nil || param == nil || *param == nil {
		return
	}
	params := (*param).(*ConvertCodexResponseToClaudeParams)
	events := params.DeferredStreamEvents
	params.DeferredStreamEvents = nil
	for _, event := range events {
		for _, translated := range ConvertCodexResponseToClaude(context.Background(), "", originalRequestRawJSON, nil, event, param) {
			output.WriteString(translated)
		}
	}
}

// appendClaudeStreamError 将 Codex 流式错误映射为 Claude Code 可消费的 SSE error 事件。
func appendClaudeStreamError(output *strings.Builder, rootResult gjson.Result) {
	errorResult := rootResult.Get("error")
	errType := strings.TrimSpace(errorResult.Get("type").String())
	if errType == "" {
		errType = strings.TrimSpace(rootResult.Get("error_type").String())
	}
	if errType == "" {
		errType = "api_error"
	}

	code := strings.TrimSpace(errorResult.Get("code").String())
	message := strings.TrimSpace(errorResult.Get("message").String())
	if message == "" {
		message = strings.TrimSpace(rootResult.Get("message").String())
	}
	if message == "" {
		message = code
	}
	if message == "" {
		message = errType
	}

	if code == "cyber_policy" || errType == "invalid_request" {
		errType = "invalid_request_error"
	}

	payload := `{"type":"error","error":{"type":"api_error","message":""}}`
	payload, _ = sjson.Set(payload, "error.type", errType)
	payload, _ = sjson.Set(payload, "error.message", message)
	appendClaudeSSEEvent(output, "error", payload)
}

// ConvertCodexResponseToClaudeNonStream converts a non-streaming Codex response to a non-streaming Claude Code response.
// This function processes the complete Codex response and transforms it into a single Claude Code-compatible
// JSON response. It handles message content, tool calls, reasoning content, and usage metadata, combining all
// the information into a single response that matches the Claude Code API format.
func ConvertCodexResponseToClaudeNonStream(_ context.Context, _ string, originalRequestRawJSON, _ []byte, rawJSON []byte, _ *any) string {
	revNames := buildReverseMapFromClaudeOriginalShortToOriginal(originalRequestRawJSON)

	rootResult := gjson.ParseBytes(rawJSON)
	typeStr := rootResult.Get("type").String()
	if typeStr != "response.completed" && typeStr != "response.incomplete" {
		return ""
	}

	responseData := rootResult.Get("response")
	if !responseData.Exists() {
		return ""
	}

	out := `{"id":"","type":"message","role":"assistant","model":"","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}`
	out, _ = sjson.Set(out, "id", responseData.Get("id").String())
	out, _ = sjson.Set(out, "model", responseData.Get("model").String())
	inputTokens, outputTokens, cachedTokens := extractResponsesUsage(responseData.Get("usage"))
	out, _ = sjson.Set(out, "usage.input_tokens", inputTokens)
	out, _ = sjson.Set(out, "usage.output_tokens", outputTokens)
	if cachedTokens > 0 {
		out, _ = sjson.Set(out, "usage.cache_read_input_tokens", cachedTokens)
	}

	hasCompletedToolCall := false

	if output := responseData.Get("output"); output.Exists() && output.IsArray() {
		output.ForEach(func(_, item gjson.Result) bool {
			switch item.Get("type").String() {
			case "reasoning":
				thinkingBuilder := strings.Builder{}
				signature := item.Get("encrypted_content").String()
				if summary := item.Get("summary"); summary.Exists() {
					if summary.IsArray() {
						summary.ForEach(func(_, part gjson.Result) bool {
							if txt := part.Get("text"); txt.Exists() {
								thinkingBuilder.WriteString(txt.String())
							} else {
								thinkingBuilder.WriteString(part.String())
							}
							return true
						})
					} else {
						thinkingBuilder.WriteString(summary.String())
					}
				}
				if thinkingBuilder.Len() == 0 {
					if content := item.Get("content"); content.Exists() {
						if content.IsArray() {
							content.ForEach(func(_, part gjson.Result) bool {
								if txt := part.Get("text"); txt.Exists() {
									thinkingBuilder.WriteString(txt.String())
								} else {
									thinkingBuilder.WriteString(part.String())
								}
								return true
							})
						} else {
							thinkingBuilder.WriteString(content.String())
						}
					}
				}
				if thinkingBuilder.Len() > 0 || signature != "" {
					block := `{"type":"thinking","thinking":""}`
					block, _ = sjson.Set(block, "thinking", thinkingBuilder.String())
					if signature != "" {
						block, _ = sjson.Set(block, "signature", signature)
					}
					out, _ = sjson.SetRaw(out, "content.-1", block)
				}
			case "message":
				if content := item.Get("content"); content.Exists() {
					if content.IsArray() {
						content.ForEach(func(_, part gjson.Result) bool {
							if part.Get("type").String() == "output_text" {
								text := part.Get("text").String()
								if text != "" {
									block := `{"type":"text","text":""}`
									block, _ = sjson.Set(block, "text", text)
									out, _ = sjson.SetRaw(out, "content.-1", block)
								}
							}
							return true
						})
					} else {
						text := content.String()
						if text != "" {
							block := `{"type":"text","text":""}`
							block, _ = sjson.Set(block, "text", text)
							out, _ = sjson.SetRaw(out, "content.-1", block)
						}
					}
				}
			case "function_call":
				if !shouldEmitNonStreamToolUse(typeStr, item) {
					return true
				}
				hasCompletedToolCall = true
				name := item.Get("name").String()
				if original, ok := revNames[name]; ok {
					name = original
				}

				toolBlock := `{"type":"tool_use","id":"","name":"","input":{}}`
				toolBlock, _ = sjson.Set(toolBlock, "id", shortenCodexCallIDIfNeeded(util.SanitizeClaudeToolID(item.Get("call_id").String())))
				toolBlock, _ = sjson.Set(toolBlock, "name", name)
				inputRaw := "{}"
				if argsStr := item.Get("arguments").String(); argsStr != "" && gjson.Valid(argsStr) {
					argsJSON := gjson.Parse(argsStr)
					if argsJSON.IsObject() {
						inputRaw = argsJSON.Raw
					}
				}
				toolBlock, _ = sjson.SetRaw(toolBlock, "input", inputRaw)
				out, _ = sjson.SetRaw(out, "content.-1", toolBlock)
			}
			return true
		})
	}

	out, _ = sjson.Set(out, "stop_reason", mapCodexStopReasonToClaude(codexStopReason(responseData), hasCompletedToolCall))
	out = setClaudeStopSequence(out, "stop_sequence", responseData)

	return out
}

// shouldEmitNonStreamToolUse 只允许完整响应里的合法工具参数转成 Claude tool_use。
func shouldEmitNonStreamToolUse(responseType string, item gjson.Result) bool {
	if responseType != "response.completed" {
		return false
	}
	args := item.Get("arguments").String()
	if args == "" || !gjson.Valid(args) {
		return false
	}
	return gjson.Parse(args).IsObject()
}

// codexStopReason 统一提取 Codex 完成原因，优先保留显式 stop_sequence 与 incomplete reason。
func codexStopReason(responseData gjson.Result) string {
	if stopReason := responseData.Get("stop_reason"); stopReason.Exists() && stopReason.String() != "" {
		if stopReason.String() == "stop" && codexStopSequence(responseData).String() != "" {
			return "stop_sequence"
		}
		return stopReason.String()
	}
	if reason := responseData.Get("incomplete_details.reason"); reason.Exists() && reason.String() != "" {
		return reason.String()
	}
	if codexStopSequence(responseData).String() != "" {
		return "stop_sequence"
	}
	return ""
}

// mapCodexStopReasonToClaude 将 Codex/OpenAI finish reason 映射为 Claude 兼容 stop_reason。
func mapCodexStopReasonToClaude(stopReason string, hasCompletedToolCall bool) string {
	if hasCompletedToolCall && canReportClaudeToolUse(stopReason) {
		return "tool_use"
	}

	switch stopReason {
	case "", "stop", "completed":
		return "end_turn"
	case "max_tokens", "max_output_tokens":
		return "max_tokens"
	case "tool_use", "tool_calls", "function_call":
		return "tool_use"
	case "end_turn", "stop_sequence", "pause_turn", "refusal", "model_context_window_exceeded":
		return stopReason
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

// canReportClaudeToolUse 只在完成原因兼容工具完成语义时允许 tool_use 覆盖。
func canReportClaudeToolUse(stopReason string) bool {
	switch stopReason {
	case "", "stop", "completed", "stop_sequence", "tool_use", "tool_calls", "function_call":
		return true
	default:
		return false
	}
}

// codexStopSequence 封装 stop_sequence 读取，确保 stream 与 non-stream 走同一字段。
func codexStopSequence(responseData gjson.Result) gjson.Result {
	return responseData.Get("stop_sequence")
}

// setClaudeStopSequence 仅在 Codex 明确返回 stop_sequence 时写入 Claude stop_sequence。
func setClaudeStopSequence(out string, path string, responseData gjson.Result) string {
	if stopSequence := codexStopSequence(responseData); stopSequence.Exists() && stopSequence.String() != "" {
		out, _ = sjson.SetRaw(out, path, stopSequence.Raw)
	}
	return out
}

func startCodexTextBlock(params *ConvertCodexResponseToClaudeParams, output *strings.Builder) {
	if params == nil || output == nil || params.TextBlockOpen {
		return
	}
	template := `{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`
	template, _ = sjson.Set(template, "index", params.BlockIndex)
	params.TextBlockOpen = true
	appendClaudeSSEEvent(output, "content_block_start", template)
}

func closeOpenTextBlock(params *ConvertCodexResponseToClaudeParams, output *strings.Builder) {
	if params == nil || output == nil || !params.TextBlockOpen {
		return
	}
	template := `{"type":"content_block_stop","index":0}`
	template, _ = sjson.Set(template, "index", params.BlockIndex)
	params.TextBlockOpen = false
	params.BlockIndex++
	appendClaudeSSEEvent(output, "content_block_stop", template)
}

func codexFunctionCallKeys(rootResult, itemResult gjson.Result) []string {
	keys := make([]string, 0, 5)
	if outputIndex := rootResult.Get("output_index"); outputIndex.Exists() {
		keys = appendUniqueCodexFunctionCallKey(keys, "output:"+outputIndex.Raw)
	}
	if callID := itemResult.Get("call_id").String(); callID != "" {
		keys = appendUniqueCodexFunctionCallKey(keys, "call:"+callID)
	}
	if callID := rootResult.Get("call_id").String(); callID != "" {
		keys = appendUniqueCodexFunctionCallKey(keys, "call:"+callID)
	}
	if itemID := itemResult.Get("id").String(); itemID != "" {
		keys = appendUniqueCodexFunctionCallKey(keys, "item:"+itemID)
	}
	if itemID := rootResult.Get("item_id").String(); itemID != "" {
		keys = appendUniqueCodexFunctionCallKey(keys, "item:"+itemID)
	}
	return keys
}

func appendUniqueCodexFunctionCallKey(keys []string, key string) []string {
	for _, existing := range keys {
		if existing == key {
			return keys
		}
	}
	return append(keys, key)
}

func codexFunctionCallForKeys(params *ConvertCodexResponseToClaudeParams, keys []string) *codexFunctionCallStream {
	if params == nil {
		return nil
	}
	for _, key := range keys {
		if call := params.FunctionCalls[key]; call != nil {
			return call
		}
	}
	return nil
}

func codexFunctionCallForEvent(params *ConvertCodexResponseToClaudeParams, rootResult, itemResult gjson.Result) *codexFunctionCallStream {
	if keys := codexFunctionCallKeys(rootResult, itemResult); len(keys) > 0 {
		return codexFunctionCallForKeys(params, keys)
	}
	if params == nil {
		return nil
	}
	return params.LastFunctionCall
}

func recordCodexFunctionCall(params *ConvertCodexResponseToClaudeParams, rootResult, itemResult gjson.Result) *codexFunctionCallStream {
	keys := codexFunctionCallKeys(rootResult, itemResult)
	call := codexFunctionCallForKeys(params, keys)
	if call == nil {
		call = &codexFunctionCallStream{BlockIndex: -1}
		params.FunctionCallQueue = append(params.FunctionCallQueue, call)
	}
	addCodexFunctionCallAliases(params, call, keys)
	params.LastFunctionCall = call
	return call
}

func addCodexFunctionCallAliases(params *ConvertCodexResponseToClaudeParams, call *codexFunctionCallStream, keys []string) {
	if params == nil || call == nil {
		return
	}
	if params.FunctionCalls == nil {
		params.FunctionCalls = make(map[string]*codexFunctionCallStream)
	}
	for _, key := range keys {
		params.FunctionCalls[key] = call
	}
}

func updateCodexFunctionCallIdentity(params *ConvertCodexResponseToClaudeParams, call *codexFunctionCallStream, rootResult, itemResult gjson.Result) {
	if call == nil {
		return
	}
	if callID := itemResult.Get("call_id").String(); callID != "" {
		call.CallID = callID
	} else if callID := rootResult.Get("call_id").String(); callID != "" {
		call.CallID = callID
	}
	if name := itemResult.Get("name").String(); name != "" {
		call.Name = name
	}
	addCodexFunctionCallAliases(params, call, codexFunctionCallKeys(rootResult, itemResult))
}

func updateCodexFunctionCallArguments(call *codexFunctionCallStream, arguments string, delta bool) {
	if call == nil || arguments == "" {
		return
	}
	if delta {
		call.Arguments += arguments
		call.HasReceivedArgumentsDelta = true
		return
	}
	if !call.HasReceivedArgumentsDelta || strings.HasPrefix(arguments, call.Arguments) {
		call.Arguments = arguments
	}
}

func appendCodexFunctionCallStart(output *strings.Builder, originalRequestRawJSON []byte, call *codexFunctionCallStream) {
	if output == nil || call == nil {
		return
	}
	template := `{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","id":"","name":"","input":{}}}`
	template, _ = sjson.Set(template, "index", call.BlockIndex)
	template, _ = sjson.Set(template, "content_block.id", shortenCodexCallIDIfNeeded(util.SanitizeClaudeToolID(call.CallID)))
	template, _ = sjson.Set(template, "content_block.name", resolveCodexClaudeToolUseName(originalRequestRawJSON, call.Name))
	appendClaudeSSEEvent(output, "content_block_start", template)
}

func appendCodexFunctionCallArgumentDelta(output *strings.Builder, partialJSON string, blockIndex int) {
	if output == nil {
		return
	}
	template := `{"type":"content_block_delta","index":0,"delta":{"type":"input_json_delta","partial_json":""}}`
	template, _ = sjson.Set(template, "index", blockIndex)
	template, _ = sjson.Set(template, "delta.partial_json", partialJSON)
	appendClaudeSSEEvent(output, "content_block_delta", template)
}

func appendCodexFunctionCallStop(output *strings.Builder, blockIndex int) {
	if output == nil {
		return
	}
	template := `{"type":"content_block_stop","index":0}`
	template, _ = sjson.Set(template, "index", blockIndex)
	appendClaudeSSEEvent(output, "content_block_stop", template)
}

func appendCodexFunctionCallBufferedArguments(output *strings.Builder, params *ConvertCodexResponseToClaudeParams, call *codexFunctionCallStream) {
	if params == nil || call == nil || params.ActiveFunctionCall != call || !call.Started || call.Closed || call.EmittedArgumentsLength >= len(call.Arguments) {
		return
	}
	appendCodexFunctionCallArgumentDelta(output, call.Arguments[call.EmittedArgumentsLength:], call.BlockIndex)
	call.EmittedArgumentsLength = len(call.Arguments)
}

func appendCodexFunctionCallQueue(output *strings.Builder, params *ConvertCodexResponseToClaudeParams, originalRequestRawJSON []byte) {
	if output == nil || params == nil {
		return
	}
	for {
		if active := params.ActiveFunctionCall; active != nil {
			appendCodexFunctionCallBufferedArguments(output, params, active)
			if !active.Done {
				return
			}
			appendCodexFunctionCallStop(output, active.BlockIndex)
			if params.BlockIndex <= active.BlockIndex {
				params.BlockIndex = active.BlockIndex + 1
			}
			active.Closed = true
			params.ActiveFunctionCall = nil
			removeCodexFunctionCallFromQueue(params, active)
		}

		for len(params.FunctionCallQueue) > 0 && params.FunctionCallQueue[0].Closed {
			params.FunctionCallQueue = params.FunctionCallQueue[1:]
		}
		if len(params.FunctionCallQueue) == 0 {
			return
		}
		call := params.FunctionCallQueue[0]
		if call.Name == "" {
			return
		}

		call.BlockIndex = params.BlockIndex
		appendCodexFunctionCallStart(output, originalRequestRawJSON, call)
		if call.EmitInitialEmptyDelta {
			appendCodexFunctionCallArgumentDelta(output, "", call.BlockIndex)
		}
		call.Started = true
		params.ActiveFunctionCall = call
		appendCodexFunctionCallBufferedArguments(output, params, call)
	}
}

func removeCodexFunctionCallFromQueue(params *ConvertCodexResponseToClaudeParams, call *codexFunctionCallStream) {
	for index, queued := range params.FunctionCallQueue {
		if queued == call {
			params.FunctionCallQueue = append(params.FunctionCallQueue[:index], params.FunctionCallQueue[index+1:]...)
			return
		}
	}
}

func appendCodexFunctionCallsFromTerminal(output *strings.Builder, params *ConvertCodexResponseToClaudeParams, originalRequestRawJSON []byte, responseData gjson.Result, completed bool) {
	if output == nil || params == nil {
		return
	}
	responseData.Get("output").ForEach(func(index, item gjson.Result) bool {
		if item.Get("type").String() != "function_call" {
			return true
		}
		keys := codexFunctionCallKeys(gjson.Result{}, item)
		if outputIndex := item.Get("output_index"); outputIndex.Exists() {
			keys = appendUniqueCodexFunctionCallKey(keys, "output:"+outputIndex.Raw)
		}
		if index.Exists() {
			keys = appendUniqueCodexFunctionCallKey(keys, "output:"+index.String())
		}
		call := codexFunctionCallForKeys(params, keys)
		if call == nil {
			call = &codexFunctionCallStream{BlockIndex: -1}
			params.FunctionCallQueue = append(params.FunctionCallQueue, call)
		}
		addCodexFunctionCallAliases(params, call, keys)
		updateCodexFunctionCallIdentity(params, call, gjson.Result{}, item)
		updateCodexFunctionCallArguments(call, item.Get("arguments").String(), false)
		call.Done = true
		return true
	})

	queuedCalls := params.FunctionCallQueue[:0]
	for _, call := range params.FunctionCallQueue {
		if call.Closed {
			continue
		}
		if call.Name == "" {
			call.Closed = true
			continue
		}
		call.Done = true
		queuedCalls = append(queuedCalls, call)
	}
	params.FunctionCallQueue = queuedCalls
	if completed && len(queuedCalls) > 0 {
		params.HasCompletedToolCall = true
	}
	appendCodexFunctionCallQueue(output, params, originalRequestRawJSON)
	clearCodexFunctionCalls(params)
}

func clearCodexFunctionCalls(params *ConvertCodexResponseToClaudeParams) {
	params.FunctionCalls = nil
	params.FunctionCallQueue = nil
	params.ActiveFunctionCall = nil
	params.LastFunctionCall = nil
}

func resolveCodexClaudeToolUseName(originalRequestRawJSON []byte, name string) string {
	if original, ok := buildReverseMapFromClaudeOriginalShortToOriginal(originalRequestRawJSON)[name]; ok {
		return original
	}
	return name
}

func extractResponsesUsage(usage gjson.Result) (int64, int64, int64) {
	if !usage.Exists() || usage.Type == gjson.Null {
		return 0, 0, 0
	}

	inputTokens := usage.Get("input_tokens").Int()
	outputTokens := usage.Get("output_tokens").Int()
	cachedTokens := usage.Get("input_tokens_details.cached_tokens").Int()

	if cachedTokens > 0 {
		if inputTokens >= cachedTokens {
			inputTokens -= cachedTokens
		} else {
			inputTokens = 0
		}
	}

	return inputTokens, outputTokens, cachedTokens
}

// buildReverseMapFromClaudeOriginalShortToOriginal builds a map[short]original from original Claude request tools.
func buildReverseMapFromClaudeOriginalShortToOriginal(original []byte) map[string]string {
	tools := gjson.GetBytes(original, "tools")
	rev := map[string]string{}
	if !tools.IsArray() {
		return rev
	}
	var names []string
	arr := tools.Array()
	for i := 0; i < len(arr); i++ {
		n := arr[i].Get("name").String()
		if n != "" {
			names = append(names, n)
		}
	}
	if len(names) > 0 {
		m := buildShortNameMap(names)
		for orig, short := range m {
			rev[short] = orig
		}
	}
	return rev
}

func ClaudeTokenCount(ctx context.Context, count int64) string {
	return fmt.Sprintf(`{"input_tokens":%d}`, count)
}

func finalizeCodexThinkingBlock(params *ConvertCodexResponseToClaudeParams, output *strings.Builder) {
	if !params.ThinkingBlockOpen {
		return
	}

	if params.ThinkingSignature != "" {
		signatureDelta := `{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":""}}`
		signatureDelta, _ = sjson.Set(signatureDelta, "index", params.BlockIndex)
		signatureDelta, _ = sjson.Set(signatureDelta, "delta.signature", params.ThinkingSignature)
		appendClaudeSSEEvent(output, "content_block_delta", signatureDelta)
	}

	contentBlockStop := `{"type":"content_block_stop","index":0}`
	contentBlockStop, _ = sjson.Set(contentBlockStop, "index", params.BlockIndex)
	appendClaudeSSEEvent(output, "content_block_stop", contentBlockStop)

	params.BlockIndex++
	params.ThinkingBlockOpen = false
}

func startCodexThinkingBlock(params *ConvertCodexResponseToClaudeParams, output *strings.Builder) {
	if params.ThinkingBlockOpen {
		return
	}
	template := `{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":""}}`
	template, _ = sjson.Set(template, "index", params.BlockIndex)
	params.ThinkingBlockOpen = true
	appendClaudeSSEEvent(output, "content_block_start", template)
}

func appendCodexThinkingDelta(params *ConvertCodexResponseToClaudeParams, output *strings.Builder, text string) {
	if output == nil || text == "" {
		return
	}
	template := `{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":""}}`
	template, _ = sjson.Set(template, "index", params.BlockIndex)
	template, _ = sjson.Set(template, "delta.thinking", text)
	appendClaudeSSEEvent(output, "content_block_delta", template)
}

func finalizeCodexSignatureOnlyThinkingBlock(params *ConvertCodexResponseToClaudeParams, output *strings.Builder) {
	// 某些上游序列只在 reasoning item 里给 encrypted_content，没有 summary 事件。
	// 这里要主动补一个空的 thinking block，才能把 signature 还给 Claude 客户端。
	if strings.TrimSpace(params.ThinkingSignature) == "" {
		params.ThinkingSummarySeen = false
		return
	}
	startCodexThinkingBlock(params, output)
	finalizeCodexThinkingBlock(params, output)
}
