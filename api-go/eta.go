package main

// api/app.py 와 같은 HTTP 계약 (POST /eta, GET /health).
// 입력 변환 규칙은 FastAPI/pydantic(lax 모드)의 실제 동작을 따른다:
//   - distance_km: 숫자, 숫자 문자열(" 2 ", "1e3", "1_000"), bool(1/0). ge=0
//   - pending_orders: 정수, 정수형 float(3.0), 정수 문자열("3", "3.0", "+3", "1_0"), bool. ge=0
//   - 모르는 필드는 무시, 중복 키는 마지막 값
//   - 422 본문은 {"detail": [{"type", "loc", "msg", "input"}]}
// 다른 점: Python 은 inf/NaN/오버플로 입력에서 minutes:null 이나 500 을 내는 알려진 결함이 있다
// (api/tests 의 xfail). 이 서비스는 그 경우 테스트가 의도한 대로 422 를 돌려준다.

import (
	"bytes"
	"encoding/json"
	"errors"
	"io"
	"math"
	"math/big"
	"mime"
	"net/http"
	"regexp"
	"strconv"
	"strings"
)

const maxBodyBytes = 1 << 20

// Predictor 는 (거리 km, 대기 주문 수) → 예상 시간(분) 을 계산한다.
type Predictor interface {
	Predict(distanceKm, pendingOrders float32) (float32, error)
}

type fieldError struct {
	Type  string `json:"type"`
	Loc   []any  `json:"loc"`
	Msg   string `json:"msg"`
	Input any    `json:"input"`
}

func writeJSON(w http.ResponseWriter, status int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(status)
	_ = json.NewEncoder(w).Encode(v)
}

func write422(w http.ResponseWriter, errs ...fieldError) {
	writeJSON(w, http.StatusUnprocessableEntity, map[string]any{"detail": errs})
}

// NewHandler 는 라우팅(404/405 포함)까지 담은 http.Handler 를 만든다.
func NewHandler(p Predictor, model string) http.Handler {
	routes := map[string]map[string]http.HandlerFunc{
		"/health": {http.MethodGet: func(w http.ResponseWriter, r *http.Request) {
			writeJSON(w, http.StatusOK, map[string]string{"status": "ok", "model": model})
		}},
		"/eta": {http.MethodPost: func(w http.ResponseWriter, r *http.Request) {
			handleEta(w, r, p, model)
		}},
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		methods, ok := routes[r.URL.Path]
		if trimmed := strings.TrimSuffix(r.URL.Path, "/"); !ok && trimmed != r.URL.Path {
			if _, ok := routes[trimmed]; ok { // Starlette 처럼 /health/ → /health 로 보낸다
				u := *r.URL
				u.Path = trimmed
				http.Redirect(w, r, u.RequestURI(), http.StatusTemporaryRedirect)
				return
			}
		}
		if !ok {
			writeJSON(w, http.StatusNotFound, map[string]string{"detail": "Not Found"})
			return
		}
		h, ok := methods[r.Method]
		if !ok {
			allow := make([]string, 0, len(methods))
			for m := range methods {
				allow = append(allow, m)
			}
			w.Header().Set("Allow", strings.Join(allow, ", "))
			writeJSON(w, http.StatusMethodNotAllowed, map[string]string{"detail": "Method Not Allowed"})
			return
		}
		h(w, r)
	})
}

func handleEta(w http.ResponseWriter, r *http.Request, p Predictor, model string) {
	body, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		writeJSON(w, http.StatusRequestEntityTooLarge, map[string]string{"detail": "Request body too large"})
		return
	}
	distance, orders, errs := parseEtaRequest(r.Header.Get("Content-Type"), body)
	if len(errs) > 0 {
		write422(w, errs...)
		return
	}

	minutes, err := p.Predict(float32(distance), float32(orders))
	if err != nil {
		writeJSON(w, http.StatusInternalServerError, map[string]string{"detail": "Internal Server Error"})
		return
	}
	m := float64(minutes)
	if math.IsNaN(m) || math.IsInf(m, 0) {
		// float32 범위를 넘는 입력 등. JSON 은 inf/NaN 을 표현할 수 없으므로 null 대신 422.
		write422(w, fieldError{Type: "value_error", Loc: []any{"body"},
			Msg: "Value error, prediction is not a finite number (input out of model range)", Input: json.RawMessage(body)})
		return
	}
	writeJSON(w, http.StatusOK, etaResponse{Minutes: pyFloat(m), Model: model})
}

type etaResponse struct {
	Minutes pyFloat `json:"minutes"`
	Model   string  `json:"model"`
}

// pyFloat 는 Python 의 float repr 과 같은 모양으로 JSON 에 쓴다 (22 → 22.0, 3.6e19 → 3.6e+19).
// 그래야 받는 쪽에서 minutes 가 항상 float 로 읽힌다.
type pyFloat float64

func (f pyFloat) MarshalJSON() ([]byte, error) {
	v := float64(f)
	e := strconv.FormatFloat(v, 'e', -1, 64) // 가장 짧은 표현, 예: 2.2e+01
	exp, _ := strconv.Atoi(e[strings.IndexByte(e, 'e')+1:])
	if v != 0 && (exp < -4 || exp >= 16) {
		return []byte(e), nil
	}
	s := strconv.FormatFloat(v, 'f', -1, 64)
	if !strings.ContainsAny(s, ".") {
		s += ".0"
	}
	return []byte(s), nil
}

func isJSONContentType(ct string) bool {
	mt, _, err := mime.ParseMediaType(ct)
	if err != nil {
		return false
	}
	main, sub, ok := strings.Cut(mt, "/")
	return ok && main == "application" && (sub == "json" || strings.HasSuffix(sub, "+json"))
}

// parseEtaRequest 는 본문을 검증해 (distance_km, pending_orders) 를 돌려준다.
func parseEtaRequest(contentType string, body []byte) (float64, float64, []fieldError) {
	if len(body) == 0 {
		return 0, 0, []fieldError{{Type: "missing", Loc: []any{"body"}, Msg: "Field required", Input: nil}}
	}
	if !isJSONContentType(contentType) {
		return 0, 0, []fieldError{{Type: "model_attributes_type", Loc: []any{"body"},
			Msg: "Input should be a valid dictionary or object to extract fields from", Input: string(body)}}
	}

	dec := json.NewDecoder(bytes.NewReader(body))
	dec.UseNumber()
	var v any
	if err := dec.Decode(&v); err != nil {
		return 0, 0, []fieldError{jsonInvalid(err, len(body))}
	}
	if _, err := dec.Token(); !errors.Is(err, io.EOF) { // 객체 뒤에 남은 데이터
		return 0, 0, []fieldError{{Type: "json_invalid", Loc: []any{"body", dec.InputOffset()},
			Msg: "JSON decode error", Input: map[string]any{}}}
	}
	obj, ok := v.(map[string]any)
	if !ok {
		return 0, 0, []fieldError{{Type: "model_attributes_type", Loc: []any{"body"},
			Msg: "Input should be a valid dictionary or object to extract fields from", Input: v}}
	}

	var errs []fieldError
	distance, e1 := parseDistance(obj)
	if e1 != nil {
		errs = append(errs, *e1)
	}
	orders, e2 := parsePendingOrders(obj)
	if e2 != nil {
		errs = append(errs, *e2)
	}
	return distance, orders, errs
}

func jsonInvalid(err error, n int) fieldError {
	pos := int64(n)
	var se *json.SyntaxError
	if errors.As(err, &se) && se.Offset > 0 {
		pos = se.Offset - 1 // 문제가 된 문자의 위치 (Python json 과 같은 기준)
	}
	return fieldError{Type: "json_invalid", Loc: []any{"body", pos}, Msg: "JSON decode error", Input: map[string]any{}}
}

func fe(typ, field, msg string, input any) *fieldError {
	return &fieldError{Type: typ, Loc: []any{"body", field}, Msg: msg, Input: input}
}

// Python float() 문자열 문법 (밑줄 구분자 허용, 16진수 불가). inf/nan 은 따로 처리한다.
var floatStrRe = regexp.MustCompile(`^[+-]?(\d(_?\d)*(\.(\d(_?\d)*)?)?|\.\d(_?\d)*)([eE][+-]?\d(_?\d)*)?$`)
var nonFiniteStrRe = regexp.MustCompile(`(?i)^[+-]?(inf|infinity|nan)$`)

func parseDistance(obj map[string]any) (float64, *fieldError) {
	const name = "distance_km"
	raw, ok := obj[name]
	if !ok {
		return 0, fe("missing", name, "Field required", obj)
	}
	var f float64
	switch v := raw.(type) {
	case bool:
		if v {
			f = 1
		}
	case json.Number:
		var err error
		if f, err = strconv.ParseFloat(v.String(), 64); err != nil && !errors.Is(err, strconv.ErrRange) {
			return 0, fe("float_parsing", name, "Input should be a valid number, unable to parse string as a number", raw)
		}
	case string:
		s := strings.TrimSpace(v)
		if nonFiniteStrRe.MatchString(s) {
			return 0, fe("finite_number", name, "Input should be a finite number", raw)
		}
		if !floatStrRe.MatchString(s) {
			return 0, fe("float_parsing", name, "Input should be a valid number, unable to parse string as a number", raw)
		}
		f, _ = strconv.ParseFloat(strings.ReplaceAll(s, "_", ""), 64)
	default: // null, 배열, 객체
		return 0, fe("float_type", name, "Input should be a valid number", raw)
	}
	if math.IsInf(f, 0) || math.IsNaN(f) {
		return 0, fe("finite_number", name, "Input should be a finite number", raw)
	}
	if f < 0 {
		return 0, fe("greater_than_equal", name, "Input should be greater than or equal to 0", raw)
	}
	return f, nil
}

var intStrRe = regexp.MustCompile(`^[+-]?\d(_?\d)*(\.0+)?$`)

func parsePendingOrders(obj map[string]any) (float64, *fieldError) {
	const name = "pending_orders"
	raw, ok := obj[name]
	if !ok {
		return 0, fe("missing", name, "Field required", obj)
	}
	n := new(big.Int)
	switch v := raw.(type) {
	case bool:
		if v {
			n.SetInt64(1)
		}
	case json.Number:
		s := v.String()
		if !strings.ContainsAny(s, ".eE") { // 정수 리터럴: 크기 제한 없음 (Python int 와 같다)
			n.SetString(s, 10)
			break
		}
		f, err := strconv.ParseFloat(s, 64)
		if err != nil || math.Abs(f) >= 1<<63 {
			return 0, fe("int_parsing_size", name, "Unable to parse input string as an integer, exceeded maximum size", raw)
		}
		if f != math.Trunc(f) {
			return 0, fe("int_from_float", name, "Input should be a valid integer, got a number with a fractional part", raw)
		}
		n.SetInt64(int64(f))
	case string:
		s := strings.TrimSpace(v)
		if !intStrRe.MatchString(s) {
			return 0, fe("int_parsing", name, "Input should be a valid integer, unable to parse string as an integer", raw)
		}
		s, _, _ = strings.Cut(strings.ReplaceAll(s, "_", ""), ".")
		n.SetString(strings.TrimPrefix(s, "+"), 10)
	default: // null, 배열, 객체
		return 0, fe("int_type", name, "Input should be a valid integer", raw)
	}
	if n.Sign() < 0 {
		return 0, fe("greater_than_equal", name, "Input should be greater than or equal to 0", raw)
	}
	f, _ := new(big.Float).SetInt(n).Float64()
	return f, nil
}
