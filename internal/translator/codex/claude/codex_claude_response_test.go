package claude

import (
	"context"
	"strings"
	"testing"

	"github.com/tidwall/gjson"
)

func TestConvertCodexResponseToClaude_StreamThinkingIncludesSignature(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_123\",\"model\":\"gpt-5\"}}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Let me think\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}"),
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_123\"}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	startFound := false
	signatureDeltaFound := false
	stopFound := false

	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			switch data.Get("type").String() {
			case "content_block_start":
				if data.Get("content_block.type").String() == "thinking" {
					startFound = true
					if data.Get("content_block.signature").Exists() {
						t.Fatalf("thinking start block should NOT have signature field when signature is unknown: %s", line)
					}
				}
			case "content_block_delta":
				if data.Get("delta.type").String() == "signature_delta" {
					signatureDeltaFound = true
					if got := data.Get("delta.signature").String(); got != "enc_sig_123" {
						t.Fatalf("unexpected signature delta: %q", got)
					}
				}
			case "content_block_stop":
				stopFound = true
			}
		}
	}

	if !startFound {
		t.Fatal("expected thinking content_block_start event")
	}
	if !signatureDeltaFound {
		t.Fatal("expected signature_delta event for thinking block")
	}
	if !stopFound {
		t.Fatal("expected content_block_stop event for thinking block")
	}
}

func TestConvertCodexResponseToClaude_StreamCyberPolicyError(t *testing.T) {
	ctx := context.Background()
	var param any

	outputs := ConvertCodexResponseToClaude(ctx, "", []byte(`{"messages":[]}`), nil, []byte(`data: {"type":"error","error":{"type":"invalid_request","code":"cyber_policy","message":"This content was flagged for possible cybersecurity risk.","param":null},"sequence_number":3}`), &param)
	if len(outputs) != 1 {
		t.Fatalf("expected one error chunk, got %d: %q", len(outputs), outputs)
	}

	payload, ok := firstClaudeStreamPayloadForEvent(outputs[0], "error")
	if !ok {
		t.Fatalf("missing error event payload: %q", outputs[0])
	}
	if got := payload.Get("type").String(); got != "error" {
		t.Fatalf("type = %q, want error. Payload: %s", got, payload.Raw)
	}
	if got := payload.Get("error.type").String(); got != "invalid_request_error" {
		t.Fatalf("error.type = %q, want invalid_request_error. Payload: %s", got, payload.Raw)
	}
	if got := payload.Get("error.message").String(); got != "This content was flagged for possible cybersecurity risk." {
		t.Fatalf("error.message = %q. Payload: %s", got, payload.Raw)
	}
}

func TestConvertCodexResponseToClaude_StreamErrorTypeFallbackMessage(t *testing.T) {
	ctx := context.Background()
	var param any

	outputs := ConvertCodexResponseToClaude(ctx, "", []byte(`{"messages":[]}`), nil, []byte(`data: {"type":"error","error":{},"error_type":"overloaded_error"}`), &param)
	if len(outputs) != 1 {
		t.Fatalf("expected one error chunk, got %d: %q", len(outputs), outputs)
	}

	payload, ok := firstClaudeStreamPayloadForEvent(outputs[0], "error")
	if !ok {
		t.Fatalf("missing error event payload: %q", outputs[0])
	}
	if got := payload.Get("error.type").String(); got != "overloaded_error" {
		t.Fatalf("error.type = %q, want overloaded_error. Payload: %s", got, payload.Raw)
	}
	if got := payload.Get("error.message").String(); got != "overloaded_error" {
		t.Fatalf("error.message = %q, want overloaded_error. Payload: %s", got, payload.Raw)
	}
}

func TestConvertCodexResponseToClaude_StreamThinkingWithoutReasoningItemOmitsSignatureField(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Let me think\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}"),
		[]byte("data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	thinkingStartFound := false
	thinkingStopFound := false
	signatureDeltaFound := false

	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if data.Get("type").String() == "content_block_start" && data.Get("content_block.type").String() == "thinking" {
				thinkingStartFound = true
				if data.Get("content_block.signature").Exists() {
					t.Fatalf("thinking start block should NOT have signature field without encrypted_content: %s", line)
				}
			}
			if data.Get("type").String() == "content_block_stop" && data.Get("index").Int() == 0 {
				thinkingStopFound = true
			}
			if data.Get("type").String() == "content_block_delta" && data.Get("delta.type").String() == "signature_delta" {
				signatureDeltaFound = true
			}
		}
	}

	if !thinkingStartFound {
		t.Fatal("expected thinking content_block_start event")
	}
	if !thinkingStopFound {
		t.Fatal("expected thinking content_block_stop event")
	}
	if signatureDeltaFound {
		t.Fatal("did not expect signature_delta without encrypted_content")
	}
}

func TestConvertCodexResponseToClaude_StreamThinkingKeepsOneBlockAcrossSummaryParts(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_early\"}}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"First part\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Second part\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}"),
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_final\"}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	startCount := 0
	stopCount := 0
	signatures := []string{}
	thinkingText := strings.Builder{}
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if data.Get("type").String() == "content_block_start" && data.Get("content_block.type").String() == "thinking" {
				startCount++
			}
			if data.Get("type").String() == "content_block_stop" {
				stopCount++
			}
			if data.Get("type").String() == "content_block_delta" {
				switch data.Get("delta.type").String() {
				case "thinking_delta":
					thinkingText.WriteString(data.Get("delta.thinking").String())
				case "signature_delta":
					signatures = append(signatures, data.Get("delta.signature").String())
				}
			}
		}
	}

	if startCount != 1 || stopCount != 1 {
		t.Fatalf("thinking lifecycle starts/stops = %d/%d, want 1/1", startCount, stopCount)
	}
	if got := thinkingText.String(); got != "First part\n\nSecond part" {
		t.Fatalf("thinking text = %q, want joined summary parts", got)
	}
	if len(signatures) != 1 || signatures[0] != "enc_sig_final" {
		t.Fatalf("signatures = %q, want final signature only", signatures)
	}
}

func TestConvertCodexResponseToClaude_StreamSerializesInterleavedFunctionCalls(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"tools":[{"name":"Read","input_schema":{"type":"object"}}]}`)
	var param any
	chunks := [][]byte{
		[]byte(`data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_a"},"output_index":1}`),
		[]byte(`data: {"type":"response.output_item.added","item":{"type":"function_call","call_id":"call_b","name":"Read"},"output_index":2}`),
		[]byte(`data: {"type":"response.function_call_arguments.delta","delta":"{\"file_path\":\"b\"}","output_index":2}`),
		[]byte(`data: {"type":"response.output_item.done","item":{"type":"function_call","call_id":"call_b","name":"Read","arguments":"{\"file_path\":\"b\"}"},"output_index":2}`),
		[]byte(`data: {"type":"response.function_call_arguments.delta","delta":"{\"file_path\":\"a\"}","output_index":1}`),
		[]byte(`data: {"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1},"output":[{"type":"function_call","call_id":"call_a","name":"Read","arguments":"{\"file_path\":\"a\"}","output_index":1},{"type":"function_call","call_id":"call_b","name":"Read","arguments":"{\"file_path\":\"b\"}","output_index":2}]}}`),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	type toolBlock struct {
		id        string
		name      string
		arguments string
	}
	events := collectClaudeStreamEvents(outputs)
	blocks := []toolBlock{}
	openIndex := int64(-1)
	for _, event := range events {
		index := event.Get("index").Int()
		switch event.Get("type").String() {
		case "content_block_start":
			if openIndex >= 0 {
				t.Fatalf("content block %d started while %d remains open; outputs=%q", index, openIndex, outputs)
			}
			if event.Get("content_block.type").String() != "tool_use" {
				continue
			}
			openIndex = index
			blocks = append(blocks, toolBlock{
				id:   event.Get("content_block.id").String(),
				name: event.Get("content_block.name").String(),
			})
		case "content_block_delta":
			if event.Get("delta.type").String() != "input_json_delta" {
				continue
			}
			if openIndex != index || len(blocks) == 0 {
				t.Fatalf("arguments delta targets unopened block %d; outputs=%q", index, outputs)
			}
			blocks[len(blocks)-1].arguments += event.Get("delta.partial_json").String()
		case "content_block_stop":
			if openIndex != index {
				t.Fatalf("content block stop targets %d while %d is open; outputs=%q", index, openIndex, outputs)
			}
			openIndex = -1
		case "message_delta":
			if openIndex >= 0 {
				t.Fatalf("message_delta emitted while block %d remains open; outputs=%q", openIndex, outputs)
			}
			if got := event.Get("delta.stop_reason").String(); got != "tool_use" {
				t.Fatalf("stop_reason = %q, want tool_use", got)
			}
		}
	}

	if openIndex >= 0 {
		t.Fatalf("content block %d remains open; outputs=%q", openIndex, outputs)
	}
	if len(blocks) != 2 {
		t.Fatalf("tool block count = %d, want 2; blocks=%+v", len(blocks), blocks)
	}
	for i, want := range []toolBlock{
		{id: "call_a", name: "Read", arguments: `{"file_path":"a"}`},
		{id: "call_b", name: "Read", arguments: `{"file_path":"b"}`},
	} {
		if blocks[i] != want {
			t.Fatalf("tool block %d = %+v, want %+v", i, blocks[i], want)
		}
	}
}

func TestConvertCodexResponseToClaude_StreamThinkingUsesEarlyCapturedSignatureWhenDoneOmitsIt(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_early\"}}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Let me think\"}"),
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\"}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	signatureDeltaCount := 0
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if data.Get("type").String() == "content_block_delta" && data.Get("delta.type").String() == "signature_delta" {
				signatureDeltaCount++
				if got := data.Get("delta.signature").String(); got != "enc_sig_early" {
					t.Fatalf("unexpected signature delta: %q", got)
				}
			}
		}
	}

	if signatureDeltaCount != 1 {
		t.Fatalf("expected signature_delta from early-captured signature, got %d", signatureDeltaCount)
	}
}

func TestConvertCodexResponseToClaude_StreamThinkingUsesFinalDoneSignature(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_initial\"}}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.added\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_text.delta\",\"delta\":\"Let me think\"}"),
		[]byte("data: {\"type\":\"response.reasoning_summary_part.done\"}"),
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_final\"}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	signatureDeltaCount := 0
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if data.Get("type").String() == "content_block_delta" && data.Get("delta.type").String() == "signature_delta" {
				signatureDeltaCount++
				if got := data.Get("delta.signature").String(); got != "enc_sig_final" {
					t.Fatalf("unexpected signature delta: %q", got)
				}
			}
		}
	}

	if signatureDeltaCount != 1 {
		t.Fatalf("expected exactly one finalized signature_delta, got %d", signatureDeltaCount)
	}
}

func TestConvertCodexResponseToClaude_StreamThinkingSignatureOnlyBlock(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"reasoning\",\"encrypted_content\":\"enc_sig_only\"}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	startFound := false
	signatureDeltaFound := false
	stopFound := false
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			switch data.Get("type").String() {
			case "content_block_start":
				if data.Get("content_block.type").String() == "thinking" {
					startFound = true
				}
			case "content_block_delta":
				if data.Get("delta.type").String() == "signature_delta" {
					signatureDeltaFound = data.Get("delta.signature").String() == "enc_sig_only"
				}
			case "content_block_stop":
				stopFound = true
			}
		}
	}

	if !startFound || !signatureDeltaFound || !stopFound {
		t.Fatalf("signature-only reasoning should emit start/signature_delta/stop, outputs=%q", outputs)
	}
}

func TestConvertCodexResponseToClaudeNonStream_ThinkingIncludesSignature(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	response := []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_123",
			"model":"gpt-5",
			"usage":{"input_tokens":10,"output_tokens":20},
			"output":[
				{
					"type":"reasoning",
					"encrypted_content":"enc_sig_nonstream",
					"summary":[{"type":"summary_text","text":"internal reasoning"}]
				},
				{
					"type":"message",
					"content":[{"type":"output_text","text":"final answer"}]
				}
			]
		}
	}`)

	out := ConvertCodexResponseToClaudeNonStream(ctx, "", originalRequest, nil, response, nil)
	parsed := gjson.Parse(out)

	thinking := parsed.Get("content.0")
	if thinking.Get("type").String() != "thinking" {
		t.Fatalf("expected first content block to be thinking, got %s", thinking.Raw)
	}
	if got := thinking.Get("signature").String(); got != "enc_sig_nonstream" {
		t.Fatalf("expected signature to be preserved, got %q", got)
	}
	if got := thinking.Get("thinking").String(); got != "internal reasoning" {
		t.Fatalf("unexpected thinking text: %q", got)
	}
}

func TestConvertCodexResponseToClaude_StreamEmptyOutputUsesOutputItemDoneMessageFallback(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"tools":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.created\",\"response\":{\"id\":\"resp_1\",\"model\":\"gpt-5\"}}"),
		[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"message\",\"role\":\"assistant\",\"content\":[{\"type\":\"output_text\",\"text\":\"ok\"}]},\"output_index\":0}"),
		[]byte("data: {\"type\":\"response.completed\",\"response\":{\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	foundText := false
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			data := gjson.Parse(strings.TrimPrefix(line, "data: "))
			if data.Get("type").String() == "content_block_delta" && data.Get("delta.type").String() == "text_delta" && data.Get("delta.text").String() == "ok" {
				foundText = true
				break
			}
		}
		if foundText {
			break
		}
	}
	if !foundText {
		t.Fatalf("expected fallback content from response.output_item.done message; outputs=%q", outputs)
	}
}

func TestConvertCodexResponseToClaude_StreamStopReasonMapping(t *testing.T) {
	tests := []struct {
		name       string
		chunks     [][]byte
		wantReason string
	}{
		{
			name: "stop 映射为 end_turn",
			chunks: [][]byte{
				[]byte("data: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"stop\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
			},
			wantReason: "end_turn",
		},
		{
			name: "incomplete max_output_tokens 映射为 max_tokens",
			chunks: [][]byte{
				[]byte("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
			},
			wantReason: "max_tokens",
		},
		{
			name: "完整 tool call 优先于 stop",
			chunks: [][]byte{
				[]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\"}}"),
				[]byte("data: {\"type\":\"response.function_call_arguments.done\",\"arguments\":\"{}\"}"),
				[]byte("data: {\"type\":\"response.output_item.done\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\",\"arguments\":\"{}\"}}"),
				[]byte("data: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"stop\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
			},
			wantReason: "tool_use",
		},
		{
			name: "content_filter 映射为 refusal",
			chunks: [][]byte{
				[]byte("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"content_filter\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
			},
			wantReason: "refusal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			originalRequest := []byte(`{"tools":[{"name":"lookup","input_schema":{"type":"object","properties":{}}}]}`)
			var param any
			var outputs []string

			for _, chunk := range tt.chunks {
				outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
			}

			got, ok := findClaudeStreamStopReason(outputs)
			if !ok {
				t.Fatalf("did not find message_delta stop_reason; outputs=%q", outputs)
			}
			if got != tt.wantReason {
				t.Fatalf("stop_reason = %q, want %q. Outputs=%q", got, tt.wantReason, outputs)
			}
		})
	}
}

func TestConvertCodexResponseToClaude_StreamIncompleteClosesOpenTextBlock(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.content_part.added\"}"),
		[]byte("data: {\"type\":\"response.output_text.delta\",\"delta\":\"partial\"}"),
		[]byte("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	events := collectClaudeStreamEvents(outputs)
	stopIndex := indexOfClaudeEventType(events, "content_block_stop")
	deltaIndex := indexOfClaudeEventType(events, "message_delta")
	if stopIndex < 0 || deltaIndex < 0 || stopIndex > deltaIndex {
		t.Fatalf("expected content_block_stop before message_delta, outputs=%q", outputs)
	}
	if got := events[deltaIndex].Get("delta.stop_reason").String(); got != "max_tokens" {
		t.Fatalf("stop_reason = %q, want max_tokens. Outputs=%q", got, outputs)
	}
}

func TestConvertCodexResponseToClaude_StreamIncompleteToolCallDoesNotReportToolUse(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"tools":[{"name":"lookup","input_schema":{"type":"object","properties":{}}}]}`)
	var param any

	chunks := [][]byte{
		[]byte("data: {\"type\":\"response.output_item.added\",\"item\":{\"type\":\"function_call\",\"call_id\":\"call_1\",\"name\":\"lookup\"}}"),
		[]byte("data: {\"type\":\"response.function_call_arguments.delta\",\"delta\":\"{\\\"q\\\":\"}"),
		[]byte("data: {\"type\":\"response.incomplete\",\"response\":{\"incomplete_details\":{\"reason\":\"max_output_tokens\"},\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"),
	}

	var outputs []string
	for _, chunk := range chunks {
		outputs = append(outputs, ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, chunk, &param)...)
	}

	events := collectClaudeStreamEvents(outputs)
	stopIndex := indexOfClaudeEventType(events, "content_block_stop")
	deltaIndex := indexOfClaudeEventType(events, "message_delta")
	if stopIndex < 0 || deltaIndex < 0 || stopIndex > deltaIndex {
		t.Fatalf("expected tool content_block_stop before message_delta, outputs=%q", outputs)
	}
	if got := events[deltaIndex].Get("delta.stop_reason").String(); got != "max_tokens" {
		t.Fatalf("stop_reason = %q, want max_tokens. Outputs=%q", got, outputs)
	}
}

func TestConvertCodexResponseToClaude_StreamStopSequenceMapping(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	var param any

	outputs := ConvertCodexResponseToClaude(ctx, "", originalRequest, nil, []byte("data: {\"type\":\"response.completed\",\"response\":{\"stop_reason\":\"stop\",\"stop_sequence\":\"\\nEND\",\"usage\":{\"input_tokens\":1,\"output_tokens\":1}}}"), &param)
	messageDelta, ok := findClaudeStreamMessageDelta(outputs)
	if !ok {
		t.Fatalf("did not find message_delta; outputs=%q", outputs)
	}
	if got := messageDelta.Get("delta.stop_reason").String(); got != "stop_sequence" {
		t.Fatalf("stop_reason = %q, want stop_sequence. Outputs=%q", got, outputs)
	}
	if got := messageDelta.Get("delta.stop_sequence").String(); got != "\nEND" {
		t.Fatalf("stop_sequence = %q, want newline END. Outputs=%q", got, outputs)
	}
}

func TestConvertCodexResponseToClaudeNonStream_StopReasonMapping(t *testing.T) {
	tests := []struct {
		name       string
		response   []byte
		wantReason string
	}{
		{
			name: "stop 映射为 end_turn",
			response: []byte(`{
				"type":"response.completed",
				"response":{
					"id":"resp_1",
					"model":"gpt-5",
					"stop_reason":"stop",
					"usage":{"input_tokens":1,"output_tokens":1},
					"output":[]
				}
			}`),
			wantReason: "end_turn",
		},
		{
			name: "incomplete max_output_tokens 映射为 max_tokens",
			response: []byte(`{
				"type":"response.incomplete",
				"response":{
					"id":"resp_1",
					"model":"gpt-5",
					"incomplete_details":{"reason":"max_output_tokens"},
					"usage":{"input_tokens":1,"output_tokens":1},
					"output":[]
				}
			}`),
			wantReason: "max_tokens",
		},
		{
			name: "完整 tool call 优先于 stop",
			response: []byte(`{
				"type":"response.completed",
				"response":{
					"id":"resp_1",
					"model":"gpt-5",
					"stop_reason":"stop",
					"usage":{"input_tokens":1,"output_tokens":1},
					"output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{}"}]
				}
			}`),
			wantReason: "tool_use",
		},
		{
			name: "incomplete 截断 function_call 不返回 tool_use",
			response: []byte(`{
				"type":"response.incomplete",
				"response":{
					"id":"resp_1",
					"model":"gpt-5",
					"incomplete_details":{"reason":"max_output_tokens"},
					"usage":{"input_tokens":1,"output_tokens":1},
					"output":[{"type":"function_call","call_id":"call_1","name":"lookup","arguments":"{\"q\":"}]
				}
			}`),
			wantReason: "max_tokens",
		},
		{
			name: "content_filter 映射为 refusal",
			response: []byte(`{
				"type":"response.incomplete",
				"response":{
					"id":"resp_1",
					"model":"gpt-5",
					"incomplete_details":{"reason":"content_filter"},
					"usage":{"input_tokens":1,"output_tokens":1},
					"output":[]
				}
			}`),
			wantReason: "refusal",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := context.Background()
			originalRequest := []byte(`{"tools":[{"name":"lookup","input_schema":{"type":"object","properties":{}}}]}`)
			out := ConvertCodexResponseToClaudeNonStream(ctx, "", originalRequest, nil, tt.response, nil)
			parsed := gjson.Parse(out)
			if got := parsed.Get("stop_reason").String(); got != tt.wantReason {
				t.Fatalf("stop_reason = %q, want %q. Output: %s", got, tt.wantReason, out)
			}
			if tt.name == "incomplete 截断 function_call 不返回 tool_use" && parsed.Get("content.#(type==\"tool_use\")").Exists() {
				t.Fatalf("truncated function_call should not be emitted as tool_use. Output: %s", out)
			}
		})
	}
}

func TestConvertCodexResponseToClaudeNonStream_StopSequenceMapping(t *testing.T) {
	ctx := context.Background()
	originalRequest := []byte(`{"messages":[]}`)
	response := []byte(`{
		"type":"response.completed",
		"response":{
			"id":"resp_1",
			"model":"gpt-5",
			"stop_reason":"stop",
			"stop_sequence":"\nEND",
			"usage":{"input_tokens":1,"output_tokens":1},
			"output":[]
		}
	}`)

	out := ConvertCodexResponseToClaudeNonStream(ctx, "", originalRequest, nil, response, nil)
	parsed := gjson.Parse(out)
	if got := parsed.Get("stop_reason").String(); got != "stop_sequence" {
		t.Fatalf("stop_reason = %q, want stop_sequence. Output: %s", got, out)
	}
	if got := parsed.Get("stop_sequence").String(); got != "\nEND" {
		t.Fatalf("stop_sequence = %q, want newline END. Output: %s", got, out)
	}
}

func findClaudeStreamStopReason(outputs []string) (string, bool) {
	messageDelta, ok := findClaudeStreamMessageDelta(outputs)
	if !ok {
		return "", false
	}
	return messageDelta.Get("delta.stop_reason").String(), true
}

func findClaudeStreamMessageDelta(outputs []string) (gjson.Result, bool) {
	for _, data := range collectClaudeStreamEvents(outputs) {
		if data.Get("type").String() == "message_delta" {
			return data, true
		}
	}
	return gjson.Result{}, false
}

func collectClaudeStreamEvents(outputs []string) []gjson.Result {
	events := []gjson.Result{}
	for _, out := range outputs {
		for _, line := range strings.Split(out, "\n") {
			if !strings.HasPrefix(line, "data: ") {
				continue
			}
			events = append(events, gjson.Parse(strings.TrimPrefix(line, "data: ")))
		}
	}
	return events
}

func firstClaudeStreamPayloadForEvent(output string, event string) (gjson.Result, bool) {
	var currentEvent string
	for _, line := range strings.Split(output, "\n") {
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			continue
		}
		if currentEvent != event || !strings.HasPrefix(line, "data: ") {
			continue
		}
		return gjson.Parse(strings.TrimPrefix(line, "data: ")), true
	}
	return gjson.Result{}, false
}

func indexOfClaudeEventType(events []gjson.Result, eventType string) int {
	for i, event := range events {
		if event.Get("type").String() == eventType {
			return i
		}
	}
	return -1
}
