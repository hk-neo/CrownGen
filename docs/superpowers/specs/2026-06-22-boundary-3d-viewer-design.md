# Boundary 모델 3D 예측 Side-by-Side 뷰어 — 설계 문서

> 작성일: 2026-06-22
> 대상: CrownGen boundary 예측 모델 학습 결과를 웹에서 인터랙티브하게 시각화

## 1. 목적

Boundary 예측 모델(결손 치아 자리의 실린더 경계를 예측하는 PVCNN2)의 학습 결과를
웹 브라우저에서 인터랙티브하게 탐색한다. 여러 학습 런(official / voxatt / context)의
예측을 같은 val 케이스에 대해 side-by-side로 동시에 띄우고, 카메라를 동기화해
"런 간 예측 차이"를 직관적으로 비교한다. 서버 없이 정적 파일로 동작한다.

## 2. 배경 — 시각화 대상 데이터

Boundary 모델은 주변 치아 점 + 존재 마스크를 입력받아, 결손(마스킹된) 슬롯 각각에
대해 실린더 5파라미터 `(cx, cy, cz, h, r)`(중심 xy/z, 높이, 반지름)을 예측한다.
품질은 실린더를 복셀 마스크로 변환해 GT와 비교한 Dice/IoU로 측정.

사용 가능한 체크포인트(`runs2/`):

| id | 파일 | mask_mode | 비고 |
|----|------|-----------|------|
| official | boundary_official_long.pt | official | Dice 0.770(best) |
| voxatt | boundary_voxatt.pt | official + voxel attention | Dice 0.507(하락) |
| context | boundary_context.pt | context | Dice 0.756 |
| official_cont | boundary_official_cont.pt | (cont077 파인튜닝) | Dice 0.756 — 예비 |

val 데이터: `Data/processed_norm2`, split `Data/SourceC_Teeth3DS/train_val_split.json`
(`stage1_val`, 40명).

## 3. 아키텍처 & 데이터 흐름

```
체크포인트들 + val 데이터
   │  [1] scripts/viz_boundary_web_data.py  (Python, GPU 1회 실행)
   │      - 케이스별 마스킹 고정 → 모든 런에 동일 입력(공정 비교)
   │      - 런별 예측 실린더 + GT + per-case Dice/IoU
   ▼
runs2/viz/web/data.js  (window.BOUNDARY_DATA = {...})
   │  [2] 정적 웹페이지 (Three.js)
   ▼
runs2/viz/web/index.html (+ app.js, three.min.js, style.css)
   → 브라우저에서 파일 열면 바로 동작 (file:// OK)
```

## 4. 컴포넌트

### 4.1 데이터 준비 스크립트 — `scripts/viz_boundary_web_data.py`

책임: 체크포인트들을 로드해 val 케이스별 예측을 계산하고 `data.js`로 덤프.

- 각 런을 해당 `mask_mode`로 `BoundEncoder` 인스턴스화 후 가중치 로드.
- `BoundDataset`(`stage1_val`, augment=False)에서 케이스 로드.
- **마스킹 고정**: 케이스 인덱스를 시드로 `make_exist_mask` 호출 → 모든 런이
  같은 결손 슬롯을 예측하도록 보장(런 간 공정 비교의 핵심).
- 각 런 forward → 예측 실린더 `(28,5)`. 결손 슬롯만 추출.
- GT 실린더(`sample['bound']`)의 결손 슬롯만 추출.
- 컨텍스트 치아 점(present & non-missing 슬롯)은 슬롯당 균일하게 200점으로
  다운샘플링(총 JSON 크기 수십 KB 유지).
- per-case Dice/IoU: `eval_dice_iou` 복셀화 로직을 케이스 단위로 재사용.
- 출력 JSON → `data.js`(`window.BOUNDARY_DATA = {...}` 형태, file:// 로딩 호환).

`data.js` 스키마:
```json
{
  "runs": [
    {"id":"official","label":"official","ckpt":"...","mask_mode":"official","color":"#e74c3c","final_dice":0.770}
  ],
  "cases": [
    {
      "idx": 0,
      "missing": [12, 13],
      "context": {"points": [[x,y,z], ...]},
      "gt": {"12": [cx,cy,cz,h,r], "13": [...]},
      "pred": {
        "official": {"12": [...], "13": [...], "dice": 0.79, "iou": 0.66},
        "voxatt":   {"12": [...], "13": [...], "dice": 0.52, "iou": 0.35},
        "context":  {"12": [...], "13": [...], "dice": 0.76, "iou": 0.62}
      }
    }
  ]
}
```

### 4.2 정적 웹 — `runs2/viz/web/`

파일 구성: `index.html` · `app.js` · `data.js` · `style.css` · `three.min.js`(로컬 벤더링).

레이아웃:
- **상단 바**: 케이스 네비게이터(`◀ 03/40 ▶` + 드롭다운) · 결손 치아 토글(어떤 슬롯 표시) ·
  런 토글(패널 on/off, 기본 official/voxatt/context).
- **메인**: 활성 런 수 = Three.js 캔버스 수, side-by-side. 각 패널:
  - 컨텍스트 치아 점(회색 `Points`)
  - GT 실린더(초록, 반투명) — 결손 슬롯 각각 `CylinderGeometry`
  - 예측 실린더(빨강, 불투명 + 와이어프레임)
- **동기화**: 단일 `OrbitControls` 인스턴스가 구동 → `change` 이벤트마다 모든 패널의
  카메라 행렬을 복사 후 재렌더. 한 뷰를 회전하면 전체가 같이 회전.
- **하단 패널**: 현재 케이스의 런별 Dice/IoU + 치아별 파라미터 표(GT vs 예측 + Δ).

좌표계: Procrustes 표준 프레임(z=교합면 법선). 실린더는 z축 수직, `(cx,cy)` 중심,
`cz` 높이 중심, `h` 높이, `r` 반지름.

### 4.3 비교 런(기본값)

`official` · `voxatt` · `context`. 모두 HTML 토글로 on/off 가능. `official_cont`는 예비로
데이터에 포함하되 기본 off.

## 5. 에러 처리

- 체크포인트 로드 실패 / `mask_mode` 불일치 → 해당 런 스킵 + 터미널 경고, 나머지 런은 계산.
- Three.js CDN 차단 대비 → `three.min.js`를 로컬에 벤더링하여 오프라인 보장.
- 예측 `NaN`/비정상 → 해당 실린더를 렌더에서 숨김(표에서 N/A 표기).

## 6. 검증(테스트)

- 준비 스크립트 종료 전 assert: `data.js`에 N(=40) 케이스, 각 케이스에 전 활성 런의
  예측·Dice/IoU 포함, Dice∈[0,1].
- 수동: `index.html` 열기 →
  - 3패널 정상 렌더, 회전 시 동기화 확인
  - prev/next 케이스 전환 동작
  - 하단 Dice가 각 런 로그의 FINAL 값과 대략 일치(0.77 / 0.51 / 0.76) 확인

## 7. 산출물 위치

```
scripts/viz_boundary_web_data.py          # 데이터 준비
runs2/viz/web/
  ├─ index.html
  ├─ app.js
  ├─ data.js
  ├─ style.css
  └─ three.min.js                         # 로컬 벤더링
```

## 8. 범위 외(YAGNI)

- 라이브 서버 / 실시간 예측 (미리 계산으로 충분)
- 학습 곡선 탭 (후속 작업에서 별도 추가 가능)
- 메시(PLY/OBJ) 렌더 (boundary는 실린더 파라미터만 의미 있음)
```
