package chat_completions

import (
	"testing"

	"github.com/tidwall/gjson"
)

// Basic tool-call: system + user + assistant(tool_calls, no content) + tool result.
// Expects developer msg + user msg + function_call + function_call_output.
// No empty assistant message should appear between user and function_call.
func TestToolCallSimple(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "system", "content": "You are a helpful assistant."},
			{"role": "user", "content": "What is the weather in Paris?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_1",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"city\":\"Paris\"}"
						}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_1",
				"content": "sunny, 22C"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather for a city",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()
	if len(items) != 4 {
		t.Fatalf("expected 4 input items, got %d: %s", len(items), gjson.Get(result, "input").Raw)
	}

	// system -> developer
	if items[0].Get("type").String() != "message" {
		t.Errorf("item 0: expected type 'message', got '%s'", items[0].Get("type").String())
	}
	if items[0].Get("role").String() != "developer" {
		t.Errorf("item 0: expected role 'developer', got '%s'", items[0].Get("role").String())
	}

	// user
	if items[1].Get("type").String() != "message" {
		t.Errorf("item 1: expected type 'message', got '%s'", items[1].Get("type").String())
	}
	if items[1].Get("role").String() != "user" {
		t.Errorf("item 1: expected role 'user', got '%s'", items[1].Get("role").String())
	}

	// function_call, not an empty assistant msg
	if items[2].Get("type").String() != "function_call" {
		t.Errorf("item 2: expected type 'function_call', got '%s'", items[2].Get("type").String())
	}
	if items[2].Get("call_id").String() != "call_1" {
		t.Errorf("item 2: expected call_id 'call_1', got '%s'", items[2].Get("call_id").String())
	}
	if items[2].Get("name").String() != "get_weather" {
		t.Errorf("item 2: expected name 'get_weather', got '%s'", items[2].Get("name").String())
	}
	if items[2].Get("arguments").String() != `{"city":"Paris"}` {
		t.Errorf("item 2: unexpected arguments: %s", items[2].Get("arguments").String())
	}

	// function_call_output
	if items[3].Get("type").String() != "function_call_output" {
		t.Errorf("item 3: expected type 'function_call_output', got '%s'", items[3].Get("type").String())
	}
	if items[3].Get("call_id").String() != "call_1" {
		t.Errorf("item 3: expected call_id 'call_1', got '%s'", items[3].Get("call_id").String())
	}
	if items[3].Get("output").String() != "sunny, 22C" {
		t.Errorf("item 3: expected output 'sunny, 22C', got '%s'", items[3].Get("output").String())
	}
}

// Assistant has both text content and tool_calls — the message should
// be emitted (non-empty content), followed by function_call items.
func TestToolCallWithContent(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "What is the weather?"},
			{
				"role": "assistant",
				"content": "Let me check the weather for you.",
				"tool_calls": [
					{
						"id": "call_abc",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{}"
						}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_abc",
				"content": "rainy, 15C"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()
	// user + assistant(with content) + function_call + function_call_output
	if len(items) != 4 {
		t.Fatalf("expected 4 input items, got %d: %s", len(items), gjson.Get(result, "input").Raw)
	}

	if items[0].Get("role").String() != "user" {
		t.Errorf("item 0: expected role 'user', got '%s'", items[0].Get("role").String())
	}

	// assistant with content — should be kept
	if items[1].Get("type").String() != "message" {
		t.Errorf("item 1: expected type 'message', got '%s'", items[1].Get("type").String())
	}
	if items[1].Get("role").String() != "assistant" {
		t.Errorf("item 1: expected role 'assistant', got '%s'", items[1].Get("role").String())
	}
	contentParts := items[1].Get("content").Array()
	if len(contentParts) == 0 {
		t.Errorf("item 1: assistant message should have content parts")
	}

	if items[2].Get("type").String() != "function_call" {
		t.Errorf("item 2: expected type 'function_call', got '%s'", items[2].Get("type").String())
	}
	if items[2].Get("call_id").String() != "call_abc" {
		t.Errorf("item 2: expected call_id 'call_abc', got '%s'", items[2].Get("call_id").String())
	}

	if items[3].Get("type").String() != "function_call_output" {
		t.Errorf("item 3: expected type 'function_call_output', got '%s'", items[3].Get("type").String())
	}
	if items[3].Get("call_id").String() != "call_abc" {
		t.Errorf("item 3: expected call_id 'call_abc', got '%s'", items[3].Get("call_id").String())
	}
}

// Parallel tool calls: assistant invokes 3 tools at once, all call_ids
// and outputs must be translated and paired correctly.
func TestMultipleToolCalls(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Compare weather in Paris, London and Tokyo"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_paris",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"city\":\"Paris\"}"
						}
					},
					{
						"id": "call_london",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"city\":\"London\"}"
						}
					},
					{
						"id": "call_tokyo",
						"type": "function",
						"function": {
							"name": "get_weather",
							"arguments": "{\"city\":\"Tokyo\"}"
						}
					}
				]
			},
			{"role": "tool", "tool_call_id": "call_paris", "content": "sunny, 22C"},
			{"role": "tool", "tool_call_id": "call_london", "content": "cloudy, 14C"},
			{"role": "tool", "tool_call_id": "call_tokyo", "content": "humid, 28C"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()
	// user + 3 function_call + 3 function_call_output = 7
	if len(items) != 7 {
		t.Fatalf("expected 7 input items, got %d: %s", len(items), gjson.Get(result, "input").Raw)
	}

	if items[0].Get("role").String() != "user" {
		t.Errorf("item 0: expected role 'user', got '%s'", items[0].Get("role").String())
	}

	expectedCallIDs := []string{"call_paris", "call_london", "call_tokyo"}
	for i, expectedID := range expectedCallIDs {
		idx := i + 1
		if items[idx].Get("type").String() != "function_call" {
			t.Errorf("item %d: expected type 'function_call', got '%s'", idx, items[idx].Get("type").String())
		}
		if items[idx].Get("call_id").String() != expectedID {
			t.Errorf("item %d: expected call_id '%s', got '%s'", idx, expectedID, items[idx].Get("call_id").String())
		}
	}

	expectedOutputs := []string{"sunny, 22C", "cloudy, 14C", "humid, 28C"}
	for i, expectedOutput := range expectedOutputs {
		idx := i + 4
		if items[idx].Get("type").String() != "function_call_output" {
			t.Errorf("item %d: expected type 'function_call_output', got '%s'", idx, items[idx].Get("type").String())
		}
		if items[idx].Get("call_id").String() != expectedCallIDs[i] {
			t.Errorf("item %d: expected call_id '%s', got '%s'", idx, expectedCallIDs[i], items[idx].Get("call_id").String())
		}
		if items[idx].Get("output").String() != expectedOutput {
			t.Errorf("item %d: expected output '%s', got '%s'", idx, expectedOutput, items[idx].Get("output").String())
		}
	}
}

func TestMixedCustomToolsCallMatchingAndImageOutput(t *testing.T) {
	input := []byte(`{
		"messages": [
			{"role":"user","content":"Run the tools."},
			{"role":"assistant","content":null,"tool_calls":[
				{"id":"call_shared","type":"function","function":{"name":"shared","arguments":"{}"}},
				{"id":"call_custom","type":"function","function":{"name":"apply_patch","arguments":"patch"}},
				{"type":"custom","custom":{"name":"capture","input":"screen"}},
				{"id":"call_duplicate","type":"function","function":{"name":"shared","arguments":"{}"}},
				{"id":"call_duplicate","type":"custom","custom":{"name":"apply_patch","input":"duplicate"}}
			]},
			{"role":"tool","tool_call_id":"call_custom","content":"patched"},
			{"role":"tool","content":"[{\"type\":\"input_text\",\"text\":\"still text\"}]"},
			{"role":"tool","content":"[{\"type\":\"input_text\",\"text\":\"captured\"},{\"type\":\"input_image\",\"image_url\":\"data:image/png;base64,AA==\",\"detail\":\"original\"},{\"type\":\"image_url\",\"image_url\":{\"url\":\"https://example.com/generated.png\",\"detail\":\"high\"}}]"},
			{"role":"tool","tool_call_id":"call_duplicate","content":"ambiguous"},
			{"role":"tool","tool_call_id":"call_orphan","content":"orphan"}
		],
		"tools": [
			{"type":"function","function":{"name":"shared","parameters":{"type":"object"}}},
			{"type":"custom","name":"shared","description":"Same-name custom tool."},
			{"type":"custom","name":"apply_patch","description":"Apply a patch."},
			{"type":"custom","name":"capture","description":"Capture a screen."}
		],
		"tool_choice":{"type":"function","function":{"name":"apply_patch"}}
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-5.6-sol", input, true)
	items := gjson.GetBytes(out, "input").Array()
	expectedTypes := []string{
		"message",
		"function_call",
		"custom_tool_call",
		"custom_tool_call",
		"custom_tool_call_output",
		"function_call_output",
		"custom_tool_call_output",
	}
	if len(items) != len(expectedTypes) {
		t.Fatalf("expected %d input items, got %d: %s", len(expectedTypes), len(items), gjson.GetBytes(out, "input").Raw)
	}
	for i, expectedType := range expectedTypes {
		if got := items[i].Get("type").String(); got != expectedType {
			t.Fatalf("item %d: expected type %s, got %s: %s", i, expectedType, got, items[i].Raw)
		}
	}

	if got := items[1].Get("name").String(); got != "shared" {
		t.Fatalf("same-name declaration should default to function, got %s", items[1].Raw)
	}
	if got := items[2].Get("name").String(); got != "apply_patch" {
		t.Fatalf("expected inferred custom call, got %s", items[2].Raw)
	}
	if got := items[4].Get("call_id").String(); got != "call_custom" {
		t.Fatalf("expected explicit custom output match, got %s", items[4].Raw)
	}
	if got := items[5].Get("call_id").String(); got != "call_shared" {
		t.Fatalf("expected empty-ID output to use the first pending call, got %s", items[5].Raw)
	}
	if output := items[5].Get("output"); output.Type != gjson.String || output.String() != `[{"type":"input_text","text":"still text"}]` {
		t.Fatalf("expected text-only JSON output to remain a string, got %s", output.Raw)
	}
	synthesizedCallID := items[3].Get("call_id").String()
	if synthesizedCallID == "" || items[6].Get("call_id").String() != synthesizedCallID {
		t.Fatalf("expected synthesized custom call_id to match its output: %s", gjson.GetBytes(out, "input").Raw)
	}

	outputParts := items[6].Get("output").Array()
	if len(outputParts) != 3 || outputParts[0].Get("type").String() != "input_text" {
		t.Fatalf("expected structured text and image output, got %s", items[6].Get("output").Raw)
	}
	if got := outputParts[1].Get("image_url").String(); got != "data:image/png;base64,AA==" {
		t.Fatalf("expected input_image URL, got %s", outputParts[1].Raw)
	}
	if got := outputParts[1].Get("detail").String(); got != "original" {
		t.Fatalf("expected input_image detail, got %s", outputParts[1].Raw)
	}
	if got := outputParts[2].Get("image_url").String(); got != "https://example.com/generated.png" {
		t.Fatalf("expected image_url URL, got %s", outputParts[2].Raw)
	}
	if got := outputParts[2].Get("detail").String(); got != "high" {
		t.Fatalf("expected image_url detail, got %s", outputParts[2].Raw)
	}
	if got := gjson.GetBytes(out, "tool_choice.type").String(); got != "custom" {
		t.Fatalf("expected custom tool choice, got %s", gjson.GetBytes(out, "tool_choice").Raw)
	}
}

func TestCodexChatReasoningCompatAcceptsStringReasoning(t *testing.T) {
	input := []byte(`{
		"model": "gpt-5.2",
		"reasoning": " XHIGH ",
		"messages": [
			{"role": "user", "content": "hi"}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-5.2", input, true)

	if got := gjson.GetBytes(out, "reasoning.effort").String(); got != "xhigh" {
		t.Fatalf("reasoning.effort = %q, want %q: %s", got, "xhigh", string(out))
	}
}

func TestCodexChatReasoningCompatAcceptsNestedReasoningEffort(t *testing.T) {
	input := []byte(`{
		"model": "gpt-5.2",
		"reasoning": {"effort": " high "},
		"messages": [
			{"role": "user", "content": "hi"}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-5.2", input, true)

	if got := gjson.GetBytes(out, "reasoning.effort").String(); got != "high" {
		t.Fatalf("reasoning.effort = %q, want %q: %s", got, "high", string(out))
	}
}

func TestCodexChatReasoningCompatPrefersReasoningEffort(t *testing.T) {
	input := []byte(`{
		"model": "gpt-5.2",
		"reasoning_effort": "medium",
		"reasoning": "xhigh",
		"messages": [
			{"role": "user", "content": "hi"}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-5.2", input, true)

	if got := gjson.GetBytes(out, "reasoning.effort").String(); got != "medium" {
		t.Fatalf("reasoning.effort = %q, want %q: %s", got, "medium", string(out))
	}
}

// OpenClaw 风格的 user 文本 tool result 需要先归一化，再继续走标准
// function_call_output 翻译链。
func TestPseudoToolResultMessageBecomesFunctionCallOutput(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "先读文件"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call5u6VTmiKVl09FPPXrpMKxV8T",
						"type": "function",
						"function": {
							"name": "read",
							"arguments": "{\"path\":\"/tmp/demo.md\"}"
						}
					}
				]
			},
			{
				"role": "user",
				"content": "[Tool result for call5u6VTmiKVl09FPPXrpMKxV8T]: # 标题\n\n正文"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "read",
					"description": "读取文件",
					"parameters": {"type": "object", "properties": {"path": {"type": "string"}}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	items := gjson.GetBytes(out, "input").Array()
	if len(items) != 3 {
		t.Fatalf("expected 3 input items, got %d: %s", len(items), gjson.GetBytes(out, "input").Raw)
	}

	if items[1].Get("type").String() != "function_call" {
		t.Fatalf("item 1: expected function_call, got %s", items[1].Get("type").String())
	}
	if items[1].Get("call_id").String() != "call5u6VTmiKVl09FPPXrpMKxV8T" {
		t.Fatalf("item 1: unexpected call_id %s", items[1].Get("call_id").String())
	}

	if items[2].Get("type").String() != "function_call_output" {
		t.Fatalf("item 2: expected function_call_output, got %s", items[2].Get("type").String())
	}
	if items[2].Get("call_id").String() != "call5u6VTmiKVl09FPPXrpMKxV8T" {
		t.Fatalf("item 2: unexpected call_id %s", items[2].Get("call_id").String())
	}
	if items[2].Get("output").String() != "# 标题\n\n正文" {
		t.Fatalf("item 2: unexpected output %q", items[2].Get("output").String())
	}
}

func TestPseudoToolResultMessageKeepsMixedTurnOrder(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "先查一下"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_order",
						"type": "function",
						"function": {
							"name": "lookup",
							"arguments": "{}"
						}
					}
				]
			},
			{"role": "user", "content": "[Tool result for call_order]: ok"},
			{"role": "assistant", "content": "查询完成"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "lookup",
					"description": "查询",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	items := gjson.GetBytes(out, "input").Array()
	if len(items) != 4 {
		t.Fatalf("expected 4 input items, got %d: %s", len(items), gjson.GetBytes(out, "input").Raw)
	}

	expected := []struct {
		itemType string
		role     string
	}{
		{itemType: "message", role: "user"},
		{itemType: "function_call"},
		{itemType: "function_call_output"},
		{itemType: "message", role: "assistant"},
	}
	for i, want := range expected {
		if got := items[i].Get("type").String(); got != want.itemType {
			t.Fatalf("item %d: expected type %s, got %s", i, want.itemType, got)
		}
		if want.role != "" && items[i].Get("role").String() != want.role {
			t.Fatalf("item %d: expected role %s, got %s", i, want.role, items[i].Get("role").String())
		}
	}
}

func TestPseudoToolResultMessageWithUnknownCallIDStaysUserMessage(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "hello"},
			{"role": "user", "content": "[Tool result for call_missing]: tool body"}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	items := gjson.GetBytes(out, "input").Array()
	if len(items) != 2 {
		t.Fatalf("expected 2 input items, got %d: %s", len(items), gjson.GetBytes(out, "input").Raw)
	}
	if items[1].Get("type").String() != "message" {
		t.Fatalf("item 1: expected message, got %s", items[1].Get("type").String())
	}
	if items[1].Get("role").String() != "user" {
		t.Fatalf("item 1: expected user role, got %s", items[1].Get("role").String())
	}
	if got := items[1].Get("content.0.text").String(); got != "[Tool result for call_missing]: tool body" {
		t.Fatalf("item 1: unexpected content %q", got)
	}
}

func TestPseudoToolResultMessageOutsidePendingWindowStaysUserMessage(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "先查一下"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_order",
						"type": "function",
						"function": {
							"name": "lookup",
							"arguments": "{}"
						}
					}
				]
			},
			{"role": "user", "content": "[Tool result for call_order]: ok"},
			{"role": "assistant", "content": "查询完成"},
			{"role": "user", "content": "[Tool result for call_order]: 这串前缀是日志格式，请帮我解释含义"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "lookup",
					"description": "查询",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	items := gjson.GetBytes(out, "input").Array()
	if len(items) != 5 {
		t.Fatalf("expected 5 input items, got %d: %s", len(items), gjson.GetBytes(out, "input").Raw)
	}
	if items[4].Get("type").String() != "message" {
		t.Fatalf("item 4: expected message, got %s", items[4].Get("type").String())
	}
	if items[4].Get("role").String() != "user" {
		t.Fatalf("item 4: expected user role, got %s", items[4].Get("role").String())
	}
	if got := items[4].Get("content.0.text").String(); got != "[Tool result for call_order]: 这串前缀是日志格式，请帮我解释含义" {
		t.Fatalf("item 4: unexpected content %q", got)
	}
}

// Regression test for #2132: tool-call-only assistant messages (content:null)
// must not produce an empty message item in the translated output.
func TestNoSpuriousEmptyAssistantMessage(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Call a tool"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_x",
						"type": "function",
						"function": {"name": "do_thing", "arguments": "{}"}
					}
				]
			},
			{"role": "tool", "tool_call_id": "call_x", "content": "done"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "do_thing",
					"description": "Do a thing",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()

	for i, item := range items {
		typ := item.Get("type").String()
		role := item.Get("role").String()
		if typ == "message" && role == "assistant" {
			contentArr := item.Get("content").Array()
			if len(contentArr) == 0 {
				t.Errorf("item %d: empty assistant message breaks call_id matching. item: %s", i, item.Raw)
			}
		}
	}

	// should be exactly: user + function_call + function_call_output
	if len(items) != 3 {
		t.Fatalf("expected 3 input items (user + function_call + function_call_output), got %d: %s", len(items), gjson.Get(result, "input").Raw)
	}
	if items[0].Get("type").String() != "message" || items[0].Get("role").String() != "user" {
		t.Errorf("item 0: expected user message")
	}
	if items[1].Get("type").String() != "function_call" {
		t.Errorf("item 1: expected function_call, got %s", items[1].Get("type").String())
	}
	if items[2].Get("type").String() != "function_call_output" {
		t.Errorf("item 2: expected function_call_output, got %s", items[2].Get("type").String())
	}
}

// Two rounds of tool calling in one conversation, with a text reply in between.
func TestMultiTurnToolCalling(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Weather in Paris?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{"id": "call_r1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Paris\"}"}}]
			},
			{"role": "tool", "tool_call_id": "call_r1", "content": "sunny"},
			{"role": "assistant", "content": "It is sunny in Paris."},
			{"role": "user", "content": "And London?"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [{"id": "call_r2", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"London\"}"}}]
			},
			{"role": "tool", "tool_call_id": "call_r2", "content": "rainy"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "get_weather",
					"description": "Get weather",
					"parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()
	// user, func_call(r1), func_output(r1), assistant text, user, func_call(r2), func_output(r2)
	if len(items) != 7 {
		t.Fatalf("expected 7 input items, got %d: %s", len(items), gjson.Get(result, "input").Raw)
	}

	for i, item := range items {
		if item.Get("type").String() == "message" && item.Get("role").String() == "assistant" {
			if len(item.Get("content").Array()) == 0 {
				t.Errorf("item %d: unexpected empty assistant message", i)
			}
		}
	}

	// round 1
	if items[1].Get("type").String() != "function_call" {
		t.Errorf("item 1: expected function_call, got %s", items[1].Get("type").String())
	}
	if items[1].Get("call_id").String() != "call_r1" {
		t.Errorf("item 1: expected call_id 'call_r1', got '%s'", items[1].Get("call_id").String())
	}
	if items[2].Get("type").String() != "function_call_output" {
		t.Errorf("item 2: expected function_call_output, got %s", items[2].Get("type").String())
	}

	// text reply between rounds
	if items[3].Get("type").String() != "message" || items[3].Get("role").String() != "assistant" {
		t.Errorf("item 3: expected assistant message, got type=%s role=%s", items[3].Get("type").String(), items[3].Get("role").String())
	}

	// round 2
	if items[5].Get("type").String() != "function_call" {
		t.Errorf("item 5: expected function_call, got %s", items[5].Get("type").String())
	}
	if items[5].Get("call_id").String() != "call_r2" {
		t.Errorf("item 5: expected call_id 'call_r2', got '%s'", items[5].Get("call_id").String())
	}
	if items[6].Get("type").String() != "function_call_output" {
		t.Errorf("item 6: expected function_call_output, got %s", items[6].Get("type").String())
	}
}

// Tool names over 64 chars get shortened, call_id stays the same.
func TestToolNameShortening(t *testing.T) {
	longName := "a_very_long_tool_name_that_exceeds_sixty_four_characters_limit_here_test"
	if len(longName) <= 64 {
		t.Fatalf("test setup error: name must be > 64 chars, got %d", len(longName))
	}

	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Do it"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_long",
						"type": "function",
						"function": {
							"name": "` + longName + `",
							"arguments": "{}"
						}
					}
				]
			},
			{"role": "tool", "tool_call_id": "call_long", "content": "ok"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "` + longName + `",
					"description": "A tool with a very long name",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()

	// find function_call
	var funcCallItem gjson.Result
	for _, item := range items {
		if item.Get("type").String() == "function_call" {
			funcCallItem = item
			break
		}
	}

	if !funcCallItem.Exists() {
		t.Fatal("no function_call item found in output")
	}

	// call_id unchanged
	if funcCallItem.Get("call_id").String() != "call_long" {
		t.Errorf("call_id changed: expected 'call_long', got '%s'", funcCallItem.Get("call_id").String())
	}

	// name must be truncated
	translatedName := funcCallItem.Get("name").String()
	if translatedName == longName {
		t.Errorf("tool name was NOT shortened: still '%s'", translatedName)
	}
	if len(translatedName) > 64 {
		t.Errorf("shortened name still > 64 chars: len=%d name='%s'", len(translatedName), translatedName)
	}
}

// content:"" (empty string, not null) should be treated the same as null.
func TestEmptyStringContent(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Do something"},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": [
					{
						"id": "call_empty",
						"type": "function",
						"function": {"name": "action", "arguments": "{}"}
					}
				]
			},
			{"role": "tool", "tool_call_id": "call_empty", "content": "result"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "action",
					"description": "An action",
					"parameters": {"type": "object", "properties": {}}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()

	for i, item := range items {
		if item.Get("type").String() == "message" && item.Get("role").String() == "assistant" {
			if len(item.Get("content").Array()) == 0 {
				t.Errorf("item %d: empty assistant message from content:\"\"", i)
			}
		}
	}

	// user + function_call + function_call_output
	if len(items) != 3 {
		t.Errorf("expected 3 input items, got %d", len(items))
	}
}

// Every function_call_output must have a matching function_call by call_id.
func TestCallIDsMatchBetweenCallAndOutput(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Multi-tool"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{"id": "id_a", "type": "function", "function": {"name": "tool_a", "arguments": "{}"}},
					{"id": "id_b", "type": "function", "function": {"name": "tool_b", "arguments": "{}"}}
				]
			},
			{"role": "tool", "tool_call_id": "id_a", "content": "res_a"},
			{"role": "tool", "tool_call_id": "id_b", "content": "res_b"}
		],
		"tools": [
			{"type": "function", "function": {"name": "tool_a", "description": "A", "parameters": {"type": "object", "properties": {}}}},
			{"type": "function", "function": {"name": "tool_b", "description": "B", "parameters": {"type": "object", "properties": {}}}}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	items := gjson.Get(result, "input").Array()

	// collect call_ids from function_call items
	callIDs := make(map[string]bool)
	for _, item := range items {
		if item.Get("type").String() == "function_call" {
			callIDs[item.Get("call_id").String()] = true
		}
	}

	for i, item := range items {
		if item.Get("type").String() == "function_call_output" {
			outID := item.Get("call_id").String()
			if !callIDs[outID] {
				t.Errorf("item %d: function_call_output has call_id '%s' with no matching function_call", i, outID)
			}
		}
	}

	// 2 calls, 2 outputs
	funcCallCount := 0
	funcOutputCount := 0
	for _, item := range items {
		switch item.Get("type").String() {
		case "function_call":
			funcCallCount++
		case "function_call_output":
			funcOutputCount++
		}
	}
	if funcCallCount != 2 {
		t.Errorf("expected 2 function_calls, got %d", funcCallCount)
	}
	if funcOutputCount != 2 {
		t.Errorf("expected 2 function_call_outputs, got %d", funcOutputCount)
	}
}

// Tools array should carry over to the Responses format output.
func TestToolsDefinitionTranslated(t *testing.T) {
	input := []byte(`{
		"model": "gpt-4o",
		"messages": [
			{"role": "user", "content": "Hi"}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "search",
					"description": "Search the web",
					"parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}
				}
			}
		]
	}`)

	out := ConvertOpenAIRequestToCodex("gpt-4o", input, true)
	result := string(out)

	tools := gjson.Get(result, "tools").Array()
	if len(tools) == 0 {
		t.Fatal("no tools found in output")
	}

	found := false
	for _, tool := range tools {
		if tool.Get("name").String() == "search" {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("tool 'search' not found in output tools: %s", gjson.Get(result, "tools").Raw)
	}
}
