package executor

import (
	"net/http"
	"time"

	internalcache "github.com/router-for-me/CLIProxyAPI/v7/internal/cache"
)

type codexReasoningReplayInvalidErr struct {
	err error
}

func (e codexReasoningReplayInvalidErr) Error() string {
	if e.err == nil {
		return ""
	}
	return e.err.Error()
}

func (e codexReasoningReplayInvalidErr) Unwrap() error {
	return e.err
}

func (e codexReasoningReplayInvalidErr) StatusCode() int {
	statusProvider, ok := e.err.(interface{ StatusCode() int })
	if !ok {
		return http.StatusBadRequest
	}
	return statusProvider.StatusCode()
}

func (e codexReasoningReplayInvalidErr) RetryAfter() *time.Duration {
	retryProvider, ok := e.err.(interface{ RetryAfter() *time.Duration })
	if !ok {
		return nil
	}
	return retryProvider.RetryAfter()
}

func (e codexReasoningReplayInvalidErr) Headers() http.Header {
	headersProvider, ok := e.err.(interface{ Headers() http.Header })
	if !ok {
		return nil
	}
	return headersProvider.Headers()
}

func (e codexReasoningReplayInvalidErr) SkipInvalidRequestBlock() bool {
	return true
}

func clearCodexReasoningReplayOnInvalidSignature(scope codexReasoningReplayScope, statusCode int, body []byte) bool {
	if !scope.valid() {
		return false
	}
	code, _, ok := codexStatusErrorClassification(statusCode, body)
	if ok && code == "thinking_signature_invalid" {
		internalcache.DeleteCodexReasoningReplayItem(scope.modelName, scope.sessionKey)
		return scope.injected
	}
	return false
}

func wrapCodexReasoningReplayInvalidErr(scope codexReasoningReplayScope, statusCode int, body []byte, err error) error {
	if err == nil {
		return nil
	}
	if !clearCodexReasoningReplayOnInvalidSignature(scope, statusCode, body) {
		return err
	}
	return codexReasoningReplayInvalidErr{err: err}
}

func wrapCodexReasoningReplayError(scope codexReasoningReplayScope, err error) error {
	if err == nil {
		return nil
	}
	statusProvider, ok := err.(interface{ StatusCode() int })
	if !ok {
		return err
	}
	return wrapCodexReasoningReplayInvalidErr(scope, statusProvider.StatusCode(), []byte(err.Error()), err)
}

func codexTerminalStreamErr(eventData []byte) (statusErr, []byte, bool) {
	body := codexTerminalStreamErrorBody(eventData)
	if len(body) == 0 || !codexTerminalStreamErrShouldHandle(body) {
		return statusErr{}, nil, false
	}
	return newCodexStatusErr(http.StatusBadRequest, body), body, true
}

func codexTerminalStreamErrShouldHandle(body []byte) bool {
	if codexTerminalErrorIsContextLength(body) || isCodexUsageLimitError(body) {
		return true
	}
	code, _, ok := codexStatusErrorClassification(http.StatusBadRequest, body)
	return ok && code == "thinking_signature_invalid"
}
