package main

// HTTP 계약 테스트. ONNX Runtime 없이 돌도록 rule 공식(10 + 3*km + 2*주문)의 가짜 Predictor 를 쓴다.
// 실제 모델 값은 training/check_parity.py 가 Python 서비스와 비교해 확인한다.

import (
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

type rulePredictor struct{}

func (rulePredictor) Predict(d, o float32) (float32, error) { return 10 + 3*d + 2*o, nil }

func do(t *testing.T, method, path, contentType, body string) (*httptest.ResponseRecorder, map[string]any) {
	t.Helper()
	req := httptest.NewRequest(method, path, strings.NewReader(body))
	if contentType != "" {
		req.Header.Set("Content-Type", contentType)
	}
	rec := httptest.NewRecorder()
	NewHandler(rulePredictor{}, "onnx").ServeHTTP(rec, req)
	var out map[string]any
	_ = json.Unmarshal(rec.Body.Bytes(), &out)
	return rec, out
}

func postEta(t *testing.T, body string) (*httptest.ResponseRecorder, map[string]any) {
	return do(t, http.MethodPost, "/eta", "application/json", body)
}

// /health 는 200 과 {"status": "ok", "model": "onnx"} 를 돌려준다.
func TestHealth(t *testing.T) {
	rec, out := do(t, http.MethodGet, "/health", "", "")
	if rec.Code != 200 || out["status"] != "ok" || out["model"] != "onnx" || len(out) != 2 {
		t.Fatalf("got %d %v", rec.Code, out)
	}
}

// 정상·경계·관대한 변환 입력은 200 과 공식값을, 응답은 minutes·model 두 필드만 가진다.
func TestEtaValid(t *testing.T) {
	cases := []struct {
		body string
		want float64
	}{
		{`{"distance_km":2,"pending_orders":3}`, 22},
		{`{"distance_km":1.5,"pending_orders":0}`, 14.5},
		{`{"distance_km":0,"pending_orders":0}`, 10},
		{`{"distance_km":-0.0,"pending_orders":0}`, 10},
		{`{"distance_km":1000,"pending_orders":10000}`, 23010},
		{`{"distance_km":0,"pending_orders":1000000000000000000000000000000}`, 2e30},
		{`{"distance_km":"2","pending_orders":"3"}`, 22},
		{`{"distance_km":" 1_0 ","pending_orders":" +3.0 "}`, 46},
		{`{"distance_km":2,"pending_orders":3.0}`, 22},
		{`{"distance_km":true,"pending_orders":false}`, 13},
		{`{"distance_km":1,"pending_orders":1,"foo":1}`, 15},
	}
	for _, c := range cases {
		rec, out := postEta(t, c.body)
		if rec.Code != 200 {
			t.Errorf("%s: status %d %s", c.body, rec.Code, rec.Body)
			continue
		}
		got, ok := out["minutes"].(float64)
		if !ok || len(out) != 2 || out["model"] != "onnx" || math.Abs(got-c.want) > 1e-6*math.Max(1, c.want) {
			t.Errorf("%s: got %v, want minutes=%v", c.body, out, c.want)
		}
	}
}

// 음수·숫자 아님·정수 아님·null·누락·비유한 입력은 422 이고 detail[].loc 가 문제 필드를 가리킨다.
func TestEtaInvalidField(t *testing.T) {
	cases := []struct {
		body string
		locs []string
	}{
		{`{"distance_km":-1,"pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":-0.01,"pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":1,"pending_orders":-1}`, []string{"pending_orders"}},
		{`{"distance_km":"abc","pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":"0x10","pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":1,"pending_orders":"many"}`, []string{"pending_orders"}},
		{`{"distance_km":1,"pending_orders":1.5}`, []string{"pending_orders"}},
		{`{"distance_km":1,"pending_orders":"3.5"}`, []string{"pending_orders"}},
		{`{"distance_km":1,"pending_orders":"3."}`, []string{"pending_orders"}},
		{`{"distance_km":1,"pending_orders":1e30}`, []string{"pending_orders"}},
		{`{"distance_km":null,"pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":1,"pending_orders":null}`, []string{"pending_orders"}},
		{`{"distance_km":[],"pending_orders":0}`, []string{"distance_km"}},
		{`{"pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":1}`, []string{"pending_orders"}},
		{`{}`, []string{"distance_km", "pending_orders"}},
		{`{"distance_km":-1,"pending_orders":-1}`, []string{"distance_km", "pending_orders"}},
		{`{"distance_km":"inf","pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":"nan","pending_orders":0}`, []string{"distance_km"}},
		{`{"distance_km":1e400,"pending_orders":0}`, []string{"distance_km"}},
	}
	for _, c := range cases {
		rec, out := postEta(t, c.body)
		if rec.Code != 422 {
			t.Errorf("%s: status %d, want 422", c.body, rec.Code)
			continue
		}
		detail, _ := out["detail"].([]any)
		var got []string
		for _, d := range detail {
			loc := d.(map[string]any)["loc"].([]any)
			if len(loc) != 2 || loc[0] != "body" {
				t.Errorf("%s: bad loc %v", c.body, loc)
				continue
			}
			got = append(got, loc[1].(string))
		}
		if strings.Join(got, ",") != strings.Join(c.locs, ",") {
			t.Errorf("%s: locs %v, want %v", c.body, got, c.locs)
		}
	}
}

// JSON 이 아니거나 객체가 아닌 본문, JSON 이 아닌 content-type 은 422 를 돌려준다.
func TestEtaMalformedBody(t *testing.T) {
	cases := []struct{ ct, body string }{
		{"application/json", `{bad`},
		{"application/json", ``},
		{"application/json", `[1, 2]`},
		{"application/json", `null`},
		{"application/json", `{"distance_km":1,"pending_orders":1} x`},
		{"application/json", `{"distance_km":NaN,"pending_orders":0}`},
		{"text/plain", `{"distance_km":1,"pending_orders":1}`},
		{"", `{"distance_km":1,"pending_orders":1}`},
	}
	for _, c := range cases {
		rec, out := do(t, http.MethodPost, "/eta", c.ct, c.body)
		if _, ok := out["detail"].([]any); rec.Code != 422 || !ok {
			t.Errorf("%q %q: got %d %s, want 422 with detail list", c.ct, c.body, rec.Code, rec.Body)
		}
	}
}

// application/*+json 과 charset 이 붙은 JSON content-type 도 받는다.
func TestEtaJSONContentTypeVariants(t *testing.T) {
	for _, ct := range []string{"application/json; charset=utf-8", "application/vnd.x+json"} {
		rec, _ := do(t, http.MethodPost, "/eta", ct, `{"distance_km":1,"pending_orders":1}`)
		if rec.Code != 200 {
			t.Errorf("%s: status %d", ct, rec.Code)
		}
	}
}

// 모델 출력이 유한하지 않으면 minutes:null 대신 422 를 돌려준다.
func TestEtaNonFinitePredictionReturns422(t *testing.T) {
	rec, _ := postEta(t, `{"distance_km":1e308,"pending_orders":0}`) // float32 로 바꾸면 inf
	if rec.Code != 422 {
		t.Fatalf("status %d %s, want 422", rec.Code, rec.Body)
	}
}

// minutes 는 Python float 처럼 항상 소수점이나 지수가 있는 JSON 숫자로 쓴다 (22 → 22.0).
func TestEtaMinutesIsFloatLiteral(t *testing.T) {
	cases := map[float64]string{22: "22.0", 14.5: "14.5", 0: "0.0", 3.6e19: "3.6e+19", 1e16: "1e+16", 1e15: "1000000000000000.0", 1e-5: "1e-05"}
	for v, want := range cases {
		b, _ := json.Marshal(etaResponse{Minutes: pyFloat(v), Model: "onnx"})
		if got := string(b); got != `{"minutes":`+want+`,"model":"onnx"}` {
			t.Errorf("%v: got %s, want minutes %s", v, got, want)
		}
	}
	rec, _ := postEta(t, `{"distance_km":2,"pending_orders":3}`)
	if !strings.Contains(rec.Body.String(), `"minutes":22.0`) {
		t.Errorf("body %s", rec.Body)
	}
}

// 깨진 JSON 의 loc 는 ["body", 문제 문자 위치] 다.
func TestEtaJSONInvalidLoc(t *testing.T) {
	_, out := postEta(t, `{bad`)
	loc := out["detail"].([]any)[0].(map[string]any)["loc"].([]any)
	if loc[0] != "body" || loc[1] != float64(1) {
		t.Errorf("loc %v, want [body 1]", loc)
	}
}

// 끝에 / 가 붙은 경로는 307 로 / 없는 경로에 보낸다.
func TestTrailingSlashRedirect(t *testing.T) {
	rec, _ := do(t, http.MethodGet, "/health/?x=1", "", "")
	if rec.Code != 307 || rec.Header().Get("Location") != "/health?x=1" {
		t.Errorf("got %d Location=%q", rec.Code, rec.Header().Get("Location"))
	}
}

// 허용하지 않는 메서드는 405 + Allow, 없는 경로는 404 를 JSON 으로 돌려준다.
func TestRouting(t *testing.T) {
	cases := []struct {
		method, path string
		code         int
		allow        string
	}{
		{http.MethodGet, "/eta", 405, "POST"},
		{http.MethodPut, "/eta", 405, "POST"},
		{http.MethodPost, "/health", 405, "GET"},
		{http.MethodGet, "/nope", 404, ""},
	}
	for _, c := range cases {
		rec, out := do(t, c.method, c.path, "", "")
		if rec.Code != c.code || rec.Header().Get("Allow") != c.allow || out["detail"] == nil {
			t.Errorf("%s %s: got %d allow=%q %v", c.method, c.path, rec.Code, rec.Header().Get("Allow"), out)
		}
	}
}
