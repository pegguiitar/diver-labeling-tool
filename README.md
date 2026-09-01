# Diver Labeling Tool

3D 소나 + 2D 카메라 다이버 어노테이션 툴 ([labelCloud](https://github.com/ch-sa/labelCloud) 기반).

이 가이드는 저장소 clone, 데이터셋 내려받기, 환경 설정, 첫 라벨링 세션 시작까지의 과정을 순서대로 안내합니다. **모든 단계를 순서대로** 따라 주세요.

---

## 1. 사전 준비물

계속하기 전에 아래 소프트웨어를 설치하세요.

- **Git** — https://git-scm.com
- **Miniconda** 또는 **Anaconda** — https://docs.conda.io/en/latest/miniconda.html
- Python 3.10 은 아래 설정 명령에서 자동으로 설치됩니다.

---

## 2. 저장소 Clone

이 저장소는 **공개(public)** 이므로 별도의 접근 권한이나 토큰 없이 바로 clone 할 수 있습니다. 터미널(Windows는 Anaconda Prompt)을 열고 실행:

```bash
git clone https://github.com/pegguiitar/diver-labeling-tool.git
cd diver-labeling-tool
```

---

## 3. 데이터셋 내려받기

데이터셋과 튜토리얼 영상은 Google Drive에 있습니다. 아래 링크에서 내려받으세요.

```
<관리자에게 문의: converted_labels 데이터셋 Google Drive 링크>
```

- 데이터를 압축 해제하여 **`converted_labels/` 폴더가 clone 한 저장소 폴더와 같은 부모 디렉토리 아래(형제 위치)에 오도록** 배치합니다.
- 튜토리얼 영상(mp4)을 첫 라벨링 세션 전에 시청하세요 — 툴 조작을 시연합니다.

압축 해제 후 폴더 구조는 다음과 같아야 합니다 (저장소 폴더와 `converted_labels/` 가 **형제**):

```
<부모 폴더>/
    diver-labeling-tool/        ← clone 한 저장소
        label_scene.py
        auto_track.py
        camera_panel.py
        config.ini
        labels/
        patches/
    converted_labels/           ← 데이터셋 (저장소와 형제 위치)
        Person1/
            scene_0000/
                sonar/          (3D 소나 포인트클라우드, frame_XXXXXX.bin)
                camera/         (2D 이미지, frame_XXXXXX.jpg)
                labels/         (centroid_abs labelCloud 라벨, frame_XXXXXX.json)
                pairs.jsonl
            scene_0042/
            ...
            annotations/        (툴이 생성하는 씬별 통합 JSON)
        Person2/
        Person3/
        Person4/
```

> 본인은 배정받은 **Person 폴더**(예: `Person1`)에 대해서만 작업합니다. 아래 실행 명령의 첫 인자가 이 Person 폴더입니다.

> **경로가 다른 경우:** 툴은 `converted_labels/` 를 저장소 폴더의 부모 디렉토리에서 찾습니다(`label_scene.py`, `auto_track.py` 의 `CONVERTED_DIR = DATA_DIR / "converted_labels"`). 데이터를 다른 위치에 두었다면 두 파일의 `CONVERTED_DIR` 값을 본인 환경에 맞게 수정하세요.

---

## 4. Conda 환경 생성

아래 명령으로 환경을 만들고 활성화합니다:

```bash
conda create -n labelcloud python=3.10 -y
conda activate labelcloud
pip install labelCloud
pip install "setuptools<70"
```

> `setuptools` 버전 고정은 labelCloud가 `pkg_resources`를 사용하기 때문입니다(setuptools >= 70 에서 제거됨).

---

## 5. labelCloud 패치 적용

이 툴은 설치된 labelCloud 패키지에 6가지 수정(카메라 패널 통합, 소나 좌표 보정, 프레임 동기화 훅, 라벨 저장 포맷 보존, 박스 방향 화살표, 박스 내부 점 강조)을 필요로 합니다. 아래 스크립트가 모두 자동 적용합니다:

```bash
python patches/apply_patches.py
```

다음과 같이 출력되어야 합니다:

```
  patched labelCloud/io/pointclouds/numpy.py
  patched labelCloud/view/gui.py
  patched labelCloud/control/controller.py
  patched labelCloud/io/labels/centroid.py
  patched labelCloud/model/bbox.py
  patched labelCloud/view/viewer.py
Done.
```

---

## 6. Python 경로 설정

`label_scene.py`(그리고 `auto_track.py`)를 열어 파일 상단의 `LABELCLOUD_PYTHON` 변수를 본인 conda 환경의 Python 실행 파일 경로로 수정합니다.

Windows:

```python
LABELCLOUD_PYTHON = Path(r"C:\Users\<사용자명>\miniconda3\envs\labelcloud\python.exe")
```

macOS / Linux:

```python
LABELCLOUD_PYTHON = Path("/home/<사용자명>/miniconda3/envs/labelcloud/bin/python")
```

> 정확한 경로를 찾으려면 labelcloud 환경을 활성화한 뒤 `where python`(Windows) 또는 `which python`(Mac/Linux)을 실행하세요.

---

## 7. 라벨링 툴 실행

저장소 루트 폴더에 있고 base conda 환경이 활성화된 상태인지 확인하세요. labelcloud 환경을 수동으로 활성화할 필요는 없습니다 — 스크립트가 대신 실행합니다.

명령 형식은 **`python label_scene.py <Person> <scene>`** 입니다. 첫 인자는 Person 폴더, 둘째 인자는 씬 번호 또는 씬 ID 입니다.

```bash
# Person1 의 scene_0000 라벨링
python label_scene.py Person1 0

# Person1 의 scene_0042 라벨링 (씬 ID를 직접 써도 됨)
python label_scene.py Person1 scene_0042

# 툴을 다시 열지 않고 기존 라벨만 재변환
python label_scene.py Person1 0 --convert-only
```

labelCloud 창이 열립니다. 창을 닫으면 라벨이 자동으로 통합·저장되어 **`converted_labels/<Person>/annotations/scene_XXXX.json`** 에 기록됩니다.

### (선택) 반자동 라벨 전파

`auto_track.py` 는 손으로 라벨링한 한 프레임을 같은 씬 내에서 앞뒤로 전파합니다:

```bash
python3 auto_track.py Person1 49              # 소나 + 카메라 양쪽
python3 auto_track.py Person1 49 --dry-run
python3 auto_track.py Person1 49 --modality sonar
```

---

## 8. 라벨링 워크플로우

- 왼쪽 패널: 3D 소나 포인트클라우드(높이별 색상).
- 오른쪽 아래 패널: 같은 프레임의 카메라 이미지.
- **Span Bounding Box** 로 소나 포인트클라우드에 3D 박스를 그린 뒤, Sonar Labels 드롭다운에서 클래스를 지정.
- 카메라 이미지를 클릭·드래그해 2D 박스를 그림.
- 같은 물리적 객체임을 나타내려면 소나 박스와 카메라 박스에 **같은 Link ID**를 지정.
- 프레임 이동은 `<< Previous` / `Next >>` 버튼 또는 `R` / `F` 키(← / → 키도 가능).
- 다음 프레임으로 이동하면 라벨이 자동 저장됩니다.

### 키보드 단축키

| 키 | 동작 |
|----|------|
| `R` / `F` (또는 ← / →) | 이전 / 다음 프레임 |
| `W` `A` `S` `D` | 박스 이동 (앞/왼/뒤/오른) |
| `Q` / `E` | 박스 위 / 아래 이동 |
| `Z` / `X` | **Z축** 회전 (반시계 / 시계) |
| `C` / `V` | **Y축** 회전 (반시계 / 시계) |
| `B` / `N` | **X축** 회전 (반시계 / 시계) |
| `I` / `O` | 길이(length) 증가 / 감소 |
| `K` / `L` | 너비(width) 증가 / 감소 |
| `,` / `.` | 높이(height) 증가 / 감소 |
| `T` | **가장 최근 라벨 프레임의 라벨을 현재 프레임으로 복사** (Link ID 포함) |
| `↑` / `↓` | 이전 / 다음 박스 선택 |
| `1`–`9` | 번호로 박스 선택 |
| `G` | **Re-stamp Sonar (overwrite)** — "Re-stamp Sonar (overwrite) →" 버튼과 동일 동작 |
| `Delete` | 선택한 박스 삭제 |
| `Ctrl` + `S` | 저장 |

> X·Y·Z축 회전(`Z`/`X`, `C`/`V`, `B`/`N`) 모두 오일러 각을 직접 더하는 대신, 현재 방향에 회전행렬을 합성한 뒤 다시 오일러로 분해하는 방식으로 적용됩니다. 세 축을 모두 이 방식으로 통일한 이유는, 한 축만 안전하게 처리하면 다른 축(z 등)이 예전처럼 오일러 필드를 직접 더하다가 y=±90° 근처(짐벌락)에서 실제와 다른 축을 돌리게 되기 때문입니다 — 이제 어떤 키를 누르든 항상 "박스 자신의 그 순간 로컬 축"을 기준으로 회전합니다.

> `T` 키는 현재 프레임에서 뒤로 스캔해 객체가 있는 **가장 최근 프레임**을 찾아 그 박스들(과 Link ID)을 현재 프레임에 복사합니다. 기존 박스는 지우지 않고 추가합니다.

> `G` 키는 저장 중인 씬 파일을 먼저 저장한 뒤, 현재(seed) 프레임의 소나 박스를 카메라 박스가 겹치는 동안 뒤 프레임들로 전파하되, **이미 라벨된 프레임도 덮어씁니다**(일반 Stamp 버튼과 달리 멈추지 않음). 재-시딩 상황(이미 stamp한 구간이 틀어져 다시 채워야 할 때)에 사용하세요.

### 박스 방향 표시 (Orientation Arrows)

현재 선택된(활성) 박스의 중심에서 로컬 x/y/z축을 가리키는 화살표 3개가 표시됩니다:

| 축 | 색상 |
|----|------|
| X축 (길이 방향) | 노랑 |
| Y축 (너비 방향) | 파랑 |
| Z축 (높이 방향) | 빨강 |

각 화살표 길이는 해당 축의 박스 치수에 비례합니다. X축 화살표에는 앞면(front/right face)을 표시하는 십자선이 추가로 그려져, 길이와 너비가 비슷한 박스에서도 앞뒤를 구분할 수 있습니다. (표시 여부는 labelCloud 메뉴의 **Settings > Show Orientation** 체크박스로 켜고 끌 수 있습니다.)

### 박스 안에 들어간 점 강조 (Inside-Box Point Highlight)

어떤 박스든 그 안에 포함되는 포인트클라우드 점들은 **시안(cyan)** 색으로 다시 칠해져, 각 박스가 실제로 어떤 점을 담고 있는지 한눈에 볼 수 있습니다. 박스를 이동·회전·크기 조절하면 실시간으로 갱신됩니다. (현재 프레임의 모든 박스에 적용됩니다.)

### 박스 내부 점 개수 (Points inside)

오른쪽 패널의 **Points inside** 항목에 현재 선택된 박스 안에 들어간 점의 개수가 표시됩니다. 박스를 옮기거나 크기를 바꾸면 실시간으로 갱신되며, 위의 시안 하이라이트로 강조되는 점 개수와 동일합니다. 선택된 박스가 없으면 `—` 로 표시됩니다.

---

## 9. 객체 클래스

`labels/_classes.json` 에 정의되어 있습니다:

- Fish
- Coral
- Rock
- Diver *(기본값 — 저장 전 필요 시 변경)*
- Structure
- ROV
- Calibration Board
- Unassigned

새 클래스를 추가하려면 `labels/_classes.json` 을 편집하고, `camera_panel.py` 의 `CLASS_COLORS` 에도 대응 항목을 추가하세요.

---

## 10. 출력 어노테이션 포맷

### 10.1 프레임별 라벨 파일 (`labels/frame_XXXXXX.json`)

labelCloud가 각 프레임을 **centroid_abs** 포맷으로 읽고 씁니다. 저장 시 오일러 각(`rotations`)뿐 아니라 **quaternion(오일러에서 재계산)** 과 업스트림 파이프라인이 부착한 플래그(`_body_frame`, `_axis_flip_v2`, `_personN_reflip` 등)가 **보존**됩니다:

```json
{
  "folder": "sonar",
  "filename": "frame_000351.bin",
  "path": ".../Person3/scene_0019/sonar/frame_000351.bin",
  "objects": [
    {
      "name": "Diver",
      "centroid": {"x": 1.44, "y": -0.94, "z": 0.15},
      "dimensions": {"length": 0.34, "width": 0.46, "height": 0.95},
      "rotations": {"x": 0.0, "y": 0.0, "z": 142.0},
      "quaternion": {"w": 0.3256, "x": 0.0, "y": 0.0, "z": 0.9455},
      "_body_frame": true,
      "_axis_flip_v2": true,
      "_person3_reflip": true
    }
  ]
}
```

> 패치되지 않은 labelCloud로 이미 저장되어 `quaternion`·플래그가 사라진 프레임은 소급 복원되지 않습니다(파일에 남은 정보가 없어 이어붙일 수 없음). 패치 적용 후 저장하는 프레임부터 보존됩니다.

### 10.2 씬별 통합 파일 (`annotations/scene_XXXX.json`)

프레임별 라벨은 **`converted_labels/<Person>/annotations/scene_XXXX.json`** 으로 통합됩니다. 소나 객체는 **원본 라벨의 모든 필드**(전체 `rotations` x/y/z, `quaternion`, per-annotator 플래그)를 그대로 유지하고, 그 위에 집계용 `class`·`link_id` 를 추가합니다. 카메라 객체는 정규화된 2D 박스를 가집니다. 같은 `link_id` 값은 두 센서가 본 동일한 물리적 객체를 의미합니다:

```json
{
  "person": "Person3",
  "scene_id": "scene_0019",
  "frames": {
    "351": {
      "objects": [
        {
          "name": "Diver",
          "centroid": {"x": 1.44, "y": -0.94, "z": 0.15},
          "dimensions": {"length": 0.34, "width": 0.46, "height": 0.95},
          "rotations": {"x": 0.0, "y": 0.0, "z": 142.0},
          "quaternion": {"w": 0.3256, "x": 0.0, "y": 0.0, "z": 0.9455},
          "_body_frame": true,
          "_axis_flip_v2": true,
          "_person3_reflip": true,
          "class": "Diver-sonar",
          "link_id": 1
        },
        {
          "class": "Diver-camera",
          "link_id": 1,
          "bbox_2d": {"x1": 0.80, "y1": 0.14, "x2": 1.0, "y2": 0.85}
        }
      ]
    }
  }
}
```

> **이전 포맷과의 차이:** 과거에는 소나 객체가 `rotation_z` 하나만 담고 데이터를 `uScenes/annotations/scene_XXXX.json` 에 저장했습니다. 현재는 Person 디렉토리 구조를 유지한 채 `converted_labels/<Person>/annotations/` 에 저장하며, 전체 x/y/z 회전·quaternion·플래그를 모두 보존합니다.
