// Package openai provides utilities to translate OpenAI Chat Completions
// request JSON into OpenAI Responses API request JSON.
// It supports tools, multimodal text/image inputs, and Structured Outputs.
// The package handles the conversion of OpenAI API requests into the format
// expected by the OpenAI Responses API, including proper mapping of messages,
// tools, and generation parameters.
package chat_completions

import (
	"encoding/json"
	"strconv"
	"strings"
)

const pseudoToolResultPrefix = "[Tool result for "

// ---------------------------------------------------------------------------
// Input structures (minimal – only fields actually used)
// ---------------------------------------------------------------------------

type chatReqInput struct {
	ReasoningEffort string          `json:"reasoning_effort"`
	Reasoning       json.RawMessage `json:"reasoning"`
	Messages        []chatMessage   `json:"messages"`
	Tools           []chatTool      `json:"tools"`
	ToolChoice      json.RawMessage `json:"tool_choice"`
	ResponseFormat  *chatRespFormat `json:"response_format"`
	Text            *chatTextCfg    `json:"text"`
}

type chatMessage struct {
	Role       string          `json:"role"`
	Content    json.RawMessage `json:"content"`
	ToolCalls  []chatToolCall  `json:"tool_calls"`
	ToolCallID string          `json:"tool_call_id"`
}

type chatTool struct {
	Type     string        `json:"type"`
	Name     string        `json:"name"`
	Function *chatToolFunc `json:"function"`
	// everything else kept as raw so we can pass it through untouched
	Raw json.RawMessage `json:"-"`
}

// UnmarshalJSON for chatTool: store the raw bytes too.
func (t *chatTool) UnmarshalJSON(data []byte) error {
	type alias chatTool
	var a alias
	if err := json.Unmarshal(data, &a); err != nil {
		return err
	}
	*t = chatTool(a)
	t.Raw = data
	return nil
}

type chatToolFunc struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	Parameters  json.RawMessage `json:"parameters"`
	Strict      *bool           `json:"strict"`
}

type chatToolCall struct {
	Type     string `json:"type"`
	ID       string `json:"id"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
	Custom struct {
		Name  string `json:"name"`
		Input string `json:"input"`
	} `json:"custom"`
}

type chatRespFormat struct {
	Type       string          `json:"type"`
	JSONSchema *chatJSONSchema `json:"json_schema"`
}

type chatJSONSchema struct {
	Name   string          `json:"name"`
	Strict *bool           `json:"strict"`
	Schema json.RawMessage `json:"schema"`
}

type chatTextCfg struct {
	Verbosity json.RawMessage `json:"verbosity"`
}

type chatContentPart struct {
	Type     string               `json:"type"`
	Text     string               `json:"text"`
	ImageURL *chatContentImageURL `json:"image_url"`
	File     *chatContentFile     `json:"file"`
}

type chatContentImageURL struct {
	URL    string `json:"url"`
	FileID string `json:"file_id"`
	Detail string `json:"detail"`
}

type chatContentFile struct {
	FileData string `json:"file_data"`
	FileID   string `json:"file_id"`
	FileURL  string `json:"file_url"`
	Filename string `json:"filename"`
}

type chatToolOutputPart struct {
	Type     string           `json:"type"`
	Text     string           `json:"text"`
	ImageURL json.RawMessage  `json:"image_url"`
	FileID   string           `json:"file_id"`
	Detail   string           `json:"detail"`
	File     *chatContentFile `json:"file"`
}

// ---------------------------------------------------------------------------
// ConvertOpenAIRequestToCodex – new fast implementation (Unmarshal/Marshal)
// ---------------------------------------------------------------------------

// ConvertOpenAIRequestToCodex converts an OpenAI Chat Completions request JSON
// into an OpenAI Responses API request JSON. The transformation follows the
// examples defined in docs/2.md exactly, including tools, multi-turn dialog,
// multimodal text/image handling, and Structured Outputs mapping.
//
// Parameters:
//   - modelName: The name of the model to use for the request
//   - rawJSON: The raw JSON request data from the OpenAI Chat Completions API
//   - stream: A boolean indicating if the request is for a streaming response
//
// Returns:
//   - []byte: The transformed request data in OpenAI Responses API format
func ConvertOpenAIRequestToCodex(modelName string, inputRawJSON []byte, stream bool) []byte {
	req, ok := cachedOpenAIRequest(inputRawJSON)
	if !ok {
		_ = json.Unmarshal(inputRawJSON, &req)
		PrimeOpenAIRequest(inputRawJSON)
	}
	req.Messages = normalizePseudoToolResultMessages(req.Messages)

	// Build request-local tool metadata and a shared shortening map.
	toolNames := make([]string, 0, len(req.Tools))
	seenToolNames := make(map[string]struct{}, len(req.Tools))
	customToolNames := make(map[string]struct{}, len(req.Tools))
	functionToolNames := make(map[string]struct{}, len(req.Tools))
	for _, t := range req.Tools {
		var name string
		switch t.Type {
		case "function":
			if t.Function != nil {
				name = t.Function.Name
				functionToolNames[name] = struct{}{}
			}
		case "custom":
			name = t.Name
			customToolNames[name] = struct{}{}
		}
		if name != "" {
			if _, seen := seenToolNames[name]; !seen {
				toolNames = append(toolNames, name)
				seenToolNames[name] = struct{}{}
			}
		}
	}
	for name := range functionToolNames {
		delete(customToolNames, name)
	}
	originalToolNameMap := buildShortNameMap(toolNames)

	resolveToolCall := func(toolCall chatToolCall) (callType, name, input string, valid bool) {
		switch toolCall.Type {
		case "custom":
			return "custom", toolCall.Custom.Name, toolCall.Custom.Input, true
		case "function":
			name = toolCall.Function.Name
			callType = "function"
			if _, custom := customToolNames[name]; custom {
				callType = "custom"
			}
			return callType, name, toolCall.Function.Arguments, true
		default:
			return "", "", "", false
		}
	}

	// ------------------------------------------------------------------
	// Build output map
	// ------------------------------------------------------------------
	out := map[string]any{
		"instructions":        "",
		"stream":              stream,
		"parallel_tool_calls": true,
		"include":             []string{"reasoning.encrypted_content"},
		"model":               modelName,
		"store":               false,
	}

	// reasoning
	effort := resolveCompatibleReasoningEffortForCodexChat(req)
	if effort == "" {
		effort = "medium"
	}
	out["reasoning"] = map[string]any{
		"effort":  effort,
		"summary": "auto",
	}

	// ------------------------------------------------------------------
	// Build input array
	// ------------------------------------------------------------------
	input := make([]any, 0, len(req.Messages))
	type pendingToolCall struct {
		callID       string
		sourceCallID string
		callType     string
		consumed     bool
	}
	var pendingToolCalls []pendingToolCall
	ambiguousToolCallIDs := make(map[string]struct{})

	for messageIndex, m := range req.Messages {
		role := m.Role
		switch role {
		case "tool":
			toolCallID := m.ToolCallID
			if _, ambiguous := ambiguousToolCallIDs[toolCallID]; toolCallID != "" && ambiguous {
				continue
			}

			pendingIndex := -1
			for index := range pendingToolCalls {
				pendingCall := &pendingToolCalls[index]
				if pendingCall.consumed {
					continue
				}
				if toolCallID == "" || pendingCall.sourceCallID == toolCallID || pendingCall.callID == toolCallID {
					pendingIndex = index
					break
				}
			}
			if pendingIndex < 0 {
				continue
			}

			pendingCall := &pendingToolCalls[pendingIndex]
			pendingCall.consumed = true
			outputType := "function_call_output"
			if pendingCall.callType == "custom" {
				outputType = "custom_tool_call_output"
			}
			input = append(input, map[string]any{
				"type":    outputType,
				"call_id": pendingCall.callID,
				"output":  buildToolCallOutput(m.Content),
			})

		default:
			pendingToolCalls = nil
			ambiguousToolCallIDs = make(map[string]struct{})

			displayRole := role
			if role == "system" {
				displayRole = "developer"
			}

			contentParts := buildContentParts(role, m.Content)

			msg := map[string]any{
				"type":    "message",
				"role":    displayRole,
				"content": contentParts,
			}
			// 只有 tool_calls 且没有文本内容的 assistant 空消息不能保留，
			// 否则会在 Responses 输入里插入多余 turn，破坏 call_id 对齐。
			if role != "assistant" || len(contentParts) > 0 {
				input = append(input, msg)
			}

			// Append function_call objects for assistant tool calls
			if role == "assistant" {
				callIDCounts := make(map[string]int, len(m.ToolCalls))
				usedCallIDs := make(map[string]struct{}, len(m.ToolCalls))
				for _, tc := range m.ToolCalls {
					_, _, _, valid := resolveToolCall(tc)
					if valid && tc.ID != "" {
						callIDCounts[tc.ID]++
						usedCallIDs[tc.ID] = struct{}{}
					}
				}
				for callID, count := range callIDCounts {
					if count > 1 {
						ambiguousToolCallIDs[callID] = struct{}{}
					}
				}

				for toolCallIndex, tc := range m.ToolCalls {
					callType, name, callInput, valid := resolveToolCall(tc)
					if !valid {
						continue
					}
					if _, ambiguous := ambiguousToolCallIDs[tc.ID]; tc.ID != "" && ambiguous {
						continue
					}

					callID := tc.ID
					if callID == "" {
						baseCallID := "call_missing_" + strconv.Itoa(messageIndex) + "_" + strconv.Itoa(toolCallIndex)
						callID = baseCallID
						for suffix := 1; ; suffix++ {
							if _, used := usedCallIDs[callID]; !used {
								break
							}
							callID = baseCallID + "_" + strconv.Itoa(suffix)
						}
						usedCallIDs[callID] = struct{}{}
					}
					pendingToolCalls = append(pendingToolCalls, pendingToolCall{
						callID:       callID,
						sourceCallID: tc.ID,
						callType:     callType,
					})

					if short, ok := originalToolNameMap[name]; ok {
						name = short
					} else {
						name = shortenNameIfNeeded(name)
					}
					if callType == "custom" {
						input = append(input, map[string]any{
							"type":    "custom_tool_call",
							"call_id": callID,
							"name":    name,
							"input":   callInput,
						})
					} else {
						input = append(input, map[string]any{
							"type":      "function_call",
							"call_id":   callID,
							"name":      name,
							"arguments": callInput,
						})
					}
				}
			}
		}
	}
	out["input"] = input

	// ------------------------------------------------------------------
	// response_format / text
	// ------------------------------------------------------------------
	textObj := buildTextObject(req.ResponseFormat, req.Text)
	if textObj != nil {
		out["text"] = textObj
	}

	// ------------------------------------------------------------------
	// tools
	// ------------------------------------------------------------------
	if len(req.Tools) > 0 {
		tools := make([]any, 0, len(req.Tools))
		for _, t := range req.Tools {
			if t.Type != "" && t.Type != "function" {
				// Built-in tool – pass through raw
				var v any
				_ = json.Unmarshal(t.Raw, &v)
				if t.Type == "custom" {
					if item, ok := v.(map[string]any); ok {
						name := t.Name
						if short, exists := originalToolNameMap[name]; exists {
							name = short
						} else {
							name = shortenNameIfNeeded(name)
						}
						item["name"] = name
					}
				}
				tools = append(tools, v)
				continue
			}
			if t.Type == "function" && t.Function != nil {
				item := map[string]any{
					"type": "function",
				}
				name := t.Function.Name
				if short, ok := originalToolNameMap[name]; ok {
					name = short
				} else {
					name = shortenNameIfNeeded(name)
				}
				item["name"] = name
				if t.Function.Description != "" {
					item["description"] = t.Function.Description
				}
				if len(t.Function.Parameters) > 0 {
					var params any
					_ = json.Unmarshal(t.Function.Parameters, &params)
					item["parameters"] = params
				}
				if t.Function.Strict != nil {
					item["strict"] = *t.Function.Strict
				}
				tools = append(tools, item)
			}
		}
		out["tools"] = tools
	}

	// ------------------------------------------------------------------
	// tool_choice
	// ------------------------------------------------------------------
	if len(req.ToolChoice) > 0 && string(req.ToolChoice) != "null" {
		// Determine if it's a JSON string or object
		var strVal string
		if err := json.Unmarshal(req.ToolChoice, &strVal); err == nil {
			out["tool_choice"] = strVal
		} else {
			var objVal map[string]any
			if err2 := json.Unmarshal(req.ToolChoice, &objVal); err2 == nil {
				tcType, _ := objVal["type"].(string)
				if tcType == "function" || tcType == "custom" {
					name, _ := objVal["name"].(string)
					if tcType == "function" {
						if fn, ok := objVal["function"].(map[string]any); ok {
							name, _ = fn["name"].(string)
						}
						if _, custom := customToolNames[name]; custom {
							tcType = "custom"
						}
					}
					if name != "" {
						if short, ok := originalToolNameMap[name]; ok {
							name = short
						} else {
							name = shortenNameIfNeeded(name)
						}
					}
					choice := map[string]any{"type": tcType}
					if name != "" {
						choice["name"] = name
					}
					out["tool_choice"] = choice
				} else if tcType != "" {
					out["tool_choice"] = objVal
				}
			}
		}
	}

	b, _ := json.Marshal(out)
	return b
}

// resolveCompatibleReasoningEffortForCodexChat 在 Codex Chat 入口兼容三种
// 思考档位写法：标准 `reasoning_effort`、误传的 `reasoning.effort`，以及
// 某些客户端会塞进 extra_body 的字符串 `reasoning="xhigh"`。
// 优先级保持“当前 endpoint 的原生字段优先”，避免兼容分支反向覆盖显式配置。
func resolveCompatibleReasoningEffortForCodexChat(req chatReqInput) string {
	if effort := normalizeCompatibleReasoningEffort(req.ReasoningEffort); effort != "" {
		return effort
	}
	return extractCompatibleReasoningEffortFromRaw(req.Reasoning)
}

// extractCompatibleReasoningEffortFromRaw 只做 Codex 兼容兜底：允许把
// `reasoning` 既当字符串，也当 `{effort: ...}` 对象来读。
// 其它异常结构保持旧行为，由后续链路继续处理。
func extractCompatibleReasoningEffortFromRaw(raw json.RawMessage) string {
	var text string
	if err := json.Unmarshal(raw, &text); err == nil {
		return normalizeCompatibleReasoningEffort(text)
	}

	var payload struct {
		Effort string `json:"effort"`
	}
	if err := json.Unmarshal(raw, &payload); err == nil {
		return normalizeCompatibleReasoningEffort(payload.Effort)
	}

	return ""
}

func normalizeCompatibleReasoningEffort(value string) string {
	return strings.ToLower(strings.TrimSpace(value))
}

// ---------------------------------------------------------------------------
// ConvertOpenAIRequestToCodexLegacy – original gjson/sjson implementation
// kept for equivalence testing.
// ---------------------------------------------------------------------------

// ConvertOpenAIRequestToCodexLegacy is the original implementation using
// gjson/sjson for equivalence testing against the new implementation.
func ConvertOpenAIRequestToCodexLegacy(modelName string, inputRawJSON []byte, stream bool) []byte {
	return convertOpenAIRequestToCodexLegacyImpl(modelName, inputRawJSON, stream)
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// normalizePseudoToolResultMessages 会把 OpenClaw 风格的伪装 tool result
// user 消息，在“当前仍处于待提交 tool output 的窗口”内收口成标准
// tool 消息，复用现有 function_call_output 翻译链。
func normalizePseudoToolResultMessages(messages []chatMessage) []chatMessage {
	if len(messages) == 0 {
		return nil
	}

	normalized := append([]chatMessage(nil), messages...)
	pendingCallIDs := make(map[string]struct{}, len(messages))
	for i, message := range normalized {
		if callID, output, ok := parsePseudoToolResultMessage(message, pendingCallIDs); ok {
			normalized[i].Role = "tool"
			normalized[i].ToolCallID = callID
			normalized[i].Content = marshalStringRawMessage(output)
			delete(pendingCallIDs, callID)
			continue
		}

		switch message.Role {
		case "assistant":
			clearPendingToolCallIDs(pendingCallIDs)
			for _, toolCall := range message.ToolCalls {
				if toolCall.ID != "" {
					pendingCallIDs[toolCall.ID] = struct{}{}
				}
			}
		case "tool":
			if message.ToolCallID != "" {
				delete(pendingCallIDs, message.ToolCallID)
			}
		case "user":
			clearPendingToolCallIDs(pendingCallIDs)
		}
	}
	return normalized
}

// parsePseudoToolResultMessage 只识别纯字符串 user 内容，格式固定为
// [Tool result for <call_id>]: <output>，且 call_id 必须仍处于最近一轮
// assistant tool_calls 打开的 pending 窗口中，避免把后续引用日志的
// 普通用户消息误判成 tool output。
func parsePseudoToolResultMessage(message chatMessage, pendingCallIDs map[string]struct{}) (string, string, bool) {
	if message.Role != "user" || firstNonSpaceByte(message.Content) != '"' {
		return "", "", false
	}

	contentStr, ok := unmarshalStringMessage(message.Content)
	if !ok || !strings.HasPrefix(contentStr, pseudoToolResultPrefix) {
		return "", "", false
	}

	callIDEnd := strings.Index(contentStr[len(pseudoToolResultPrefix):], "]:")
	if callIDEnd < 0 {
		return "", "", false
	}

	callIDStart := len(pseudoToolResultPrefix)
	callIDEnd += callIDStart
	callID := contentStr[callIDStart:callIDEnd]
	if callID == "" {
		return "", "", false
	}
	if _, ok := pendingCallIDs[callID]; !ok {
		return "", "", false
	}

	output := contentStr[callIDEnd+2:]
	if strings.HasPrefix(output, " ") {
		output = output[1:]
	}
	return callID, output, true
}

func clearPendingToolCallIDs(pendingCallIDs map[string]struct{}) {
	for callID := range pendingCallIDs {
		delete(pendingCallIDs, callID)
	}
}

func unmarshalStringMessage(raw json.RawMessage) (string, bool) {
	var s string
	if err := json.Unmarshal(raw, &s); err != nil {
		return "", false
	}
	return s, true
}

func marshalStringRawMessage(value string) json.RawMessage {
	raw, _ := json.Marshal(value)
	return raw
}

func buildToolCallOutput(raw json.RawMessage) any {
	var content string
	if err := json.Unmarshal(raw, &content); err == nil {
		var parts []json.RawMessage
		if json.Unmarshal([]byte(content), &parts) == nil && hasToolOutputImagePart(parts) {
			return buildToolOutputParts(parts)
		}
		return content
	}

	var parts []json.RawMessage
	if json.Unmarshal(raw, &parts) == nil {
		return buildToolOutputParts(parts)
	}
	return rawToString(raw)
}

func buildToolOutputParts(parts []json.RawMessage) []any {
	output := make([]any, 0, len(parts))
	for _, raw := range parts {
		var part chatToolOutputPart
		if err := json.Unmarshal(raw, &part); err != nil {
			output = append(output, toolOutputFallbackPart(raw))
			continue
		}

		switch part.Type {
		case "text", "input_text", "output_text":
			output = append(output, map[string]any{"type": "input_text", "text": part.Text})
		case "image_url", "input_image":
			imageURL, fileID, detail := toolOutputImageFields(part)
			if imageURL == "" && fileID == "" {
				output = append(output, toolOutputFallbackPart(raw))
				continue
			}
			item := map[string]any{"type": "input_image"}
			if imageURL != "" {
				item["image_url"] = imageURL
			}
			if fileID != "" {
				item["file_id"] = fileID
			}
			if detail != "" {
				item["detail"] = detail
			}
			output = append(output, item)
		case "file":
			if part.File == nil || (part.File.FileID == "" && part.File.FileData == "" && part.File.FileURL == "") {
				output = append(output, toolOutputFallbackPart(raw))
				continue
			}
			item := map[string]any{"type": "input_file"}
			if part.File.FileID != "" {
				item["file_id"] = part.File.FileID
			}
			if part.File.FileData != "" {
				item["file_data"] = part.File.FileData
			}
			if part.File.FileURL != "" {
				item["file_url"] = part.File.FileURL
			}
			if part.File.Filename != "" {
				item["filename"] = part.File.Filename
			}
			output = append(output, item)
		default:
			output = append(output, toolOutputFallbackPart(raw))
		}
	}
	return output
}

func hasToolOutputImagePart(parts []json.RawMessage) bool {
	for _, raw := range parts {
		var part chatToolOutputPart
		if json.Unmarshal(raw, &part) != nil || (part.Type != "image_url" && part.Type != "input_image") {
			continue
		}
		imageURL, fileID, _ := toolOutputImageFields(part)
		if imageURL != "" || fileID != "" {
			return true
		}
	}
	return false
}

func toolOutputImageFields(part chatToolOutputPart) (imageURL, fileID, detail string) {
	_ = json.Unmarshal(part.ImageURL, &imageURL)
	var nested chatContentImageURL
	if json.Unmarshal(part.ImageURL, &nested) == nil {
		if imageURL == "" {
			imageURL = nested.URL
		}
		fileID = nested.FileID
		detail = nested.Detail
	}
	if part.FileID != "" {
		fileID = part.FileID
	}
	if part.Detail != "" {
		detail = part.Detail
	}
	return imageURL, fileID, detail
}

func toolOutputFallbackPart(raw json.RawMessage) map[string]any {
	return map[string]any{"type": "input_text", "text": string(raw)}
}

func rawToString(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	// Keep behavior aligned with legacy gjson String() for null.
	if string(raw) == "null" {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	return string(raw)
}

func buildContentParts(role string, raw json.RawMessage) []any {
	parts := make([]any, 0)
	if len(raw) == 0 {
		return parts
	}

	first := firstNonSpaceByte(raw)
	switch first {
	case '"':
		var s string
		if err := json.Unmarshal(raw, &s); err != nil {
			return parts
		}
		if s == "" {
			return parts
		}
		partType := "input_text"
		if role == "assistant" {
			partType = "output_text"
		}
		parts = append(parts, map[string]any{
			"type": partType,
			"text": s,
		})
		return parts
	case '[':
	default:
		return parts
	}

	var arr []chatContentPart
	if err := json.Unmarshal(raw, &arr); err != nil {
		return parts
	}
	for _, item := range arr {
		switch item.Type {
		case "text":
			partType := "input_text"
			if role == "assistant" {
				partType = "output_text"
			}
			parts = append(parts, map[string]any{
				"type": partType,
				"text": item.Text,
			})
		case "image_url":
			if role == "user" && item.ImageURL != nil && item.ImageURL.URL != "" {
				part := map[string]any{"type": "input_image"}
				part["image_url"] = item.ImageURL.URL
				parts = append(parts, part)
			}
		case "file":
			if role == "user" && item.File != nil {
				if item.File.FileData == "" {
					continue
				}
				part := map[string]any{
					"type":      "input_file",
					"file_data": item.File.FileData,
				}
				if item.File.Filename != "" {
					part["filename"] = item.File.Filename
				}
				parts = append(parts, part)
			}
		}
	}
	return parts
}

func firstNonSpaceByte(raw json.RawMessage) byte {
	for _, b := range raw {
		switch b {
		case ' ', '\n', '\r', '\t':
			continue
		default:
			return b
		}
	}
	return 0
}

func buildTextObject(rf *chatRespFormat, tc *chatTextCfg) map[string]any {
	if rf == nil && tc == nil {
		return nil
	}

	textObj := map[string]any{}

	if rf != nil {
		format := map[string]any{}
		switch rf.Type {
		case "text":
			format["type"] = "text"
		case "json_schema":
			format["type"] = "json_schema"
			if rf.JSONSchema != nil {
				if rf.JSONSchema.Name != "" {
					format["name"] = rf.JSONSchema.Name
				}
				if rf.JSONSchema.Strict != nil {
					format["strict"] = *rf.JSONSchema.Strict
				}
				if len(rf.JSONSchema.Schema) > 0 {
					var schema any
					_ = json.Unmarshal(rf.JSONSchema.Schema, &schema)
					format["schema"] = schema
				}
			}
		}
		if len(format) > 0 {
			textObj["format"] = format
		}
	}

	if tc != nil && len(tc.Verbosity) > 0 && string(tc.Verbosity) != "null" {
		var v any
		_ = json.Unmarshal(tc.Verbosity, &v)
		textObj["verbosity"] = v
	}

	if len(textObj) == 0 {
		return nil
	}
	return textObj
}

// shortenNameIfNeeded applies the simple shortening rule for a single name.
// If the name length exceeds 64, it will try to preserve the "mcp__" prefix and last segment.
// Otherwise it truncates to 64 characters.
func shortenNameIfNeeded(name string) string {
	const limit = 64
	if len(name) <= limit {
		return name
	}
	if strings.HasPrefix(name, "mcp__") {
		// Keep prefix and last segment after '__'
		idx := strings.LastIndex(name, "__")
		if idx > 0 {
			candidate := "mcp__" + name[idx+2:]
			if len(candidate) > limit {
				return candidate[:limit]
			}
			return candidate
		}
	}
	return name[:limit]
}

// buildShortNameMap generates unique short names (<=64) for the given list of names.
// It preserves the "mcp__" prefix with the last segment when possible and ensures uniqueness
// by appending suffixes like "_1", "_2" if needed.
func buildShortNameMap(names []string) map[string]string {
	const limit = 64
	used := map[string]struct{}{}
	m := map[string]string{}

	baseCandidate := func(n string) string {
		if len(n) <= limit {
			return n
		}
		if strings.HasPrefix(n, "mcp__") {
			idx := strings.LastIndex(n, "__")
			if idx > 0 {
				cand := "mcp__" + n[idx+2:]
				if len(cand) > limit {
					cand = cand[:limit]
				}
				return cand
			}
		}
		return n[:limit]
	}

	makeUnique := func(cand string) string {
		if _, ok := used[cand]; !ok {
			return cand
		}
		base := cand
		for i := 1; ; i++ {
			suffix := "_" + strconv.Itoa(i)
			allowed := limit - len(suffix)
			if allowed < 0 {
				allowed = 0
			}
			tmp := base
			if len(tmp) > allowed {
				tmp = tmp[:allowed]
			}
			tmp = tmp + suffix
			if _, ok := used[tmp]; !ok {
				return tmp
			}
		}
	}

	for _, n := range names {
		cand := baseCandidate(n)
		uniq := makeUnique(cand)
		used[uniq] = struct{}{}
		m[n] = uniq
	}
	return m
}
