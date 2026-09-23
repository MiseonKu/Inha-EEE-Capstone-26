# 1호기 IMU 관절각 추정 설계안

## 문제 이해

이 문서는 1호기 로봇팔에서 **MPU-6050 6축 IMU 3개**만으로 상완 링크와 전완 링크의 자세를 추정하고, 그로부터 **엘보우 관련 상대각**을 계산하기 위한 구현 지향 설계안이다.

현재 사용할 IMU 배치는 다음과 같다.

- 상완 링크 IMU 2개
  - `IMU_U1`: 엘보우 축에서 7.0 cm
  - `IMU_U2`: 엘보우 축에서 14.5 cm
- 전완 링크 IMU 1개
  - `IMU_F1`: 엘보우 축에서 14.5 cm
  - 링크 중심선이 아니라 **링크 상단**에 장착되어 있음

기준 자세는 다음과 같다.

- 베이스에서 엘보우까지는 지면에 대해 수직
- 엘보우에서 말단까지는 지면에 대해 수평
- 즉, 전체적으로 90도 꺾인 기준 자세

이 자세를 **영점 자세**, **초기 캘리브레이션 자세**, **링크 기준 프레임 정의 자세**로 사용한다.

본 설계는 이미 어떤 형태로든 필터가 존재한다고 가정한다. 따라서 핵심은 다음이다.

- 필터가 낸 각 IMU 자세를 링크 자세로 해석
- 센서-링크 고정 오프셋 캘리브레이션
- 상완 2개 IMU를 이용한 상완 링크 자세 보강
- 전완 링크 자세 추정
- 상완-전완 상대 자세에서 엘보우 각 계산
- 전완 IMU의 비중심 장착에 따른 동적 가속도 보정 또는 신뢰도 저하 처리

추정 목표는 다음 순서로 본다.

1. 상완 링크 자세
2. 전완 링크 자세
3. 상완-전완 상대각
4. 엘보우 굽힘각
5. 부가적으로 상완 기울기 기반 shoulder-chain posture

---

## 무엇이 관측 가능하고 무엇이 어려운가

### 1. 안정적으로 추정 가능한 것

다음 항목은 현재 구성으로 비교적 안정적으로 추정 가능하다.

- 상완 링크의 **중력 기준 기울기**
  - 상완이 수직에서 얼마나 기울었는지
- 전완 링크의 **중력 기준 기울기**
  - 전완이 수평에서 얼마나 들렸는지 또는 처졌는지
- 상완과 전완 사이의 **상대 굽힘각**
  - 특히 엘보우가 거의 1자유도 힌지로 동작한다는 가정 아래 가장 핵심
- 기준 자세 대비 **상완/전완 링크 방향 벡터 변화**
- 상완 2개 IMU의 **일관성 오차**
  - 두 센서가 같은 강체 링크를 보고 있으므로 품질 체크에 매우 유용

### 2. 부분적으로만 추정 가능한 것

다음 항목은 제약이나 시간 범위를 두면 쓸 수 있지만 장기적으로는 불안정할 수 있다.

- shoulder-chain의 평면 내 자세 변화
  - 상완 링크 방향 벡터로부터 어느 정도는 추정 가능
  - 다만 절대 heading이 없으므로 장기적으로는 기준 평면이 떠다닐 수 있음
- 전완의 절대 방위각
  - 초기 자세 직후 짧은 구간에서는 gyro 적분으로 유지 가능
  - 장기적으로 drift 누적 가능
- 엘보우 축 주위의 상대 회전량
  - 상대 orientation 기반으로 계산 가능
  - 다만 독립 6축 필터의 yaw drift 차이가 커지면 오차가 커질 수 있음

### 3. 현재 구성으로 근본적으로 애매한 것

다음 항목은 6축 IMU만으로는 본질적으로 잘 안 보인다.

- 지구 기준 절대 yaw / heading
  - 자력계가 없으므로 절대 북쪽 기준 heading 불가
- 베이스 yaw와 링크 yaw를 장기적으로 분리 추정하는 것
- 링크 축 방향 자체의 twist
  - 예: 링크 길이축을 중심으로 돈 회전
- 전완의 절대 수평 방향의 방위각
  - 수평인 것은 보여도 어느 방위로 향하는지는 장기 drift가 생김
- 5-DoF 전체 관절의 완전한 절대 재구성
  - 지금은 shoulder-elbow-wrist chain의 자세 해석이 주 목적

### 4. 왜 기준 90도 자세가 중요한가

기준 자세는 다음 역할을 한다.

- 각 IMU의 **고정 장착 회전 오프셋** 추정
- 링크 프레임을 **현실 하드웨어에 맞게 정의**
- 상완과 전완의 링크 방향 기준축을 고정
- 엘보우 기준각의 영점 설정
- 세 IMU의 **초기 공통 heading 정렬**

하지만 중요하게도, 이 초기 정렬은 **장기 yaw drift를 근본적으로 제거하지는 못한다**.  
즉, 기준 자세는 영점과 장착 오프셋을 맞추는 데는 매우 좋지만, 6축 IMU의 절대 heading 부재 자체를 해결하지는 못한다.

---

## 좌표계와 기하 모델

### 1. 월드 프레임 `W`

기준 자세에서 다음과 같이 정의한다.

- 원점: 엘보우 축 중심
- `+Z_W`: 위쪽, 중력 반대 방향
- `+X_W`: 기준 자세에서 엘보우에서 손목 쪽으로 향하는 방향
- `+Y_W`: 오른손 좌표계를 만족하도록 정의

즉 기준 자세에서

- 상완 방향은 `+Z_W`
- 전완 방향은 `+X_W`
- 엘보우 힌지축은 `+Y_W`

### 2. 상완 링크 프레임 `U`

- 원점: 엘보우 축 중심
- `+Z_U`: 엘보우에서 숄더 방향의 링크 축
- `+Y_U`: 엘보우 힌지축
- `+X_U = Y_U × Z_U`

기준 자세에서는 `U`와 `W`가 일치하도록 잡는다.

### 3. 전완 링크 프레임 `F`

- 원점: 엘보우 축 중심
- `+X_F`: 엘보우에서 손목 방향의 링크 축
- `+Y_F`: 엘보우 힌지축
- `+Z_F = X_F × Y_F`

역시 기준 자세에서는 `F`와 `W`가 일치하도록 잡는다.

### 4. IMU 로컬 프레임 `S_i`

각 IMU에는 센서 고유 프레임이 있다.

- `S_U1`
- `S_U2`
- `S_F1`

이 프레임은 실제 PCB 장착 방향에 따라 링크 프레임과 다를 수 있다.  
따라서 각 센서마다 고정 회전 오프셋 `R_{L<-S}`가 필요하다.

### 5. 엘보우 축 정의

엘보우 축 단위벡터를 `a_E`라 두면 기준 자세에서

```text
a_E^U = a_E^F = [0, 1, 0]^T
```

이다.

실제 구현에서는 다음 두 방식 중 하나를 쓴다.

- 기본값: `a_E = +Y`
- 보정값: elbow-axis calibration으로 다시 추정

### 6. 센서 위치 벡터

엘보우 원점을 기준으로 링크 프레임에서 센서 위치를 둔다.

상완:

```text
r_U1^U = [0, 0, 0.070]^T
r_U2^U = [0, 0, 0.145]^T
```

전완:

```text
r_F1^F = [0.145, 0, h_top]^T
```

여기서 `h_top`은 전완 IMU가 링크 중심선에서 위로 얼마나 올라가 있는지이다.  
이 값은 반드시 실측해야 한다.

중요한 점:

- **정적 orientation 자체는 센서 위치와 무관**
- 그러나 accelerometer 기반 자세 보정은 센서 위치에 따라 동적 오차를 받음
- 따라서 전완 IMU의 `h_top`은 **가속도 보정/신뢰도 판단**에 반드시 써야 함

---

## 캘리브레이션 모델

### 1. 센서-링크 고정 회전 오프셋

각 센서의 필터 출력이 `R_{W<-S_i}`라고 하자.  
기준 자세에서는 `R_{W<-U_ref} = I`, `R_{W<-F_ref} = I`가 되도록 프레임을 정의했으므로,

```text
R_{U<-S_U1} = R_{W<-S_U1,ref}
R_{U<-S_U2} = R_{W<-S_U2,ref}
R_{F<-S_F1} = R_{W<-S_F1,ref}
```

가 된다.

실시간 링크 자세는 다음처럼 계산한다.

```text
R_{W<-U1} = R_{W<-S_U1} * R_{S_U1<-U}
R_{W<-U2} = R_{W<-S_U2} * R_{S_U2<-U}
R_{W<-F } = R_{W<-S_F1} * R_{S_F1<-F}
```

여기서

```text
R_{S<-L} = (R_{L<-S})^T
```

이다.

쿼터니언 표기로는 기준 자세 quaternion을 `q_ref`라 할 때

```text
q_{W<-L}(t) = q_{W<-S}(t) ⊗ q_{W<-S,ref}^{*}
```

로 구현하면 된다.

### 2. 전완 IMU의 위치 오프셋

전완 IMU가 링크 상단에 있으므로 센서가 측정하는 specific force에는 링크 원점 가속도 외에 회전에 따른 항이 더해진다.

센서 위치 벡터를 `r`이라 하면,

```text
a_sensor = a_origin + α × r + ω × (ω × r)
```

이다.

따라서 전완 IMU에서 accelerometer를 강하게 신뢰하면 다음 문제가 생긴다.

- 빠른 엘보우 운동 시 gravity 추정이 오염됨
- top offset이 클수록 오염이 커짐

실무적으로는 두 가지 중 하나를 권장한다.

1. **단순/강건 방식**
   - 전완 IMU의 accelerometer correction gain을 낮게 둔다
   - `|a|-g`가 크거나 `|ω|`가 크면 correction gain을 더 낮춘다
2. **보강 방식**
   - `α × r + ω × (ω × r)`를 추정해서 accelerometer에서 일부 차감한다
   - Uno에서는 계산량보다 안정성이 더 중요하므로 우선은 gain scheduling이 더 실용적

### 3. 기준 자세 초기화 절차

#### 정적 reference calibration

사용자에게 기준 90도 자세를 취하게 하고 2~5초간 정지시킨다.

이때 각 IMU에 대해:

- quaternion 평균
- accel 평균
- gyro 평균

을 저장한다.

저장 항목:

- `q_ref_U1`
- `q_ref_U2`
- `q_ref_F1`
- `gyro_bias_U1`
- `gyro_bias_U2`
- `gyro_bias_F1`

#### 선택적 elbow-axis calibration

기준 자세 후, 상완은 가급적 고정하고 전완만 엘보우 축 주변으로 천천히 왕복시킨다.

이 구간에서 상대 각속도

```text
ω_rel^U = R_{U<-F} ω_F^F - ω_U^U
```

의 주축을 구하면 실제 엘보우 축 방향을 얻을 수 있다.

즉,

```text
C = Σ ω_rel^U (ω_rel^U)^T
```

의 최대 고유벡터를 `a_E^U`로 둔다.

이 축이 기본 `[0,1,0]`와 크게 다르면 기계 정렬 오차가 있다는 뜻이다.

---

## 추정 수식과 로직

### 1. 상완 링크 자세

각 상완 IMU에서 보정된 링크 quaternion을 만든다.

```text
q_{W<-U}^{(1)}
q_{W<-U}^{(2)}
```

두 센서가 같은 강체를 보므로 둘의 차이각

```text
δ_U = angle( q_{W<-U}^{(1)*} ⊗ q_{W<-U}^{(2)} )
```

은 작아야 한다.

상완 최종 quaternion은 가중 평균으로 둔다.

```text
q_{W<-U} = average( q_{W<-U}^{(1)}, q_{W<-U}^{(2)} )
```

가중치는 다음 정보를 사용한다.

- accel norm이 `g`와 얼마나 가까운지
- 두 상완 IMU 사이의 불일치가 얼마나 작은지
- gyro가 너무 크지 않은지

### 2. 전완 링크 자세

전완은 센서가 1개이므로

```text
q_{W<-F} = q_{W<-S_F1} ⊗ q_{W<-S_F1,ref}^{*}
```

로 잡는다.

다만 전완 IMU는 top offset이 있으므로 accelerometer correction에 대한 신뢰도를 낮춰야 한다.

실무적으로는 다음 confidence를 둔다.

```text
c_F = f(| |a| - g |, |ω|, |α × r + ω × (ω × r)|)
```

### 3. 링크 방향 벡터 추정

기준 링크 축을 이용해서 링크 방향 벡터를 만든다.

상완:

```text
u_W = R_{W<-U} [0,0,1]^T
```

전완:

```text
f_W = R_{W<-F} [1,0,0]^T
```

엘보우 축:

```text
a_W^U = R_{W<-U} a_E^U
a_W^F = R_{W<-F} a_E^F
```

최종 힌지축은 둘을 평균해도 되고, 보통은 상완 기준이 더 안정적이므로

```text
a_W = normalize(a_W^U)
```

를 기본으로 둔다.

### 4. 1차 권장 방식: 링크 방향 벡터 기반 엘보우 각

엘보우 축에 직교하는 평면으로 두 링크 방향을 사영한다.

```text
u_p = normalize( u_W - (u_W·a_W) a_W )
f_p = normalize( f_W - (f_W·a_W) a_W )
```

그 다음 signed angle을 계산한다.

```text
θ_elbow_abs = atan2( a_W · (u_p × f_p), u_p · f_p )
```

기준 자세에서 이 값은 이론상 `+90°`이다.  
영점 자세를 0도로 만들려면

```text
θ_elbow = θ_elbow_abs - θ_ref
```

로 둔다.

여기서 `θ_ref`는 기준 자세에서 저장한 값이며 기본적으로 `+90°`이다.

이 방식의 장점:

- 링크의 실제 축을 직접 이용
- 불필요한 twist 성분을 어느 정도 무시
- 힌지 축을 명시적으로 사용
- 현재 문제에 가장 잘 맞음

### 5. 2차 백업 방식: 상대 quaternion의 swing-twist 분해

상대 자세를 계산한다.

```text
R_{U<-F} = R_{W<-U}^T R_{W<-F}
```

또는 quaternion으로

```text
q_{U<-F} = q_{W<-U}^{*} ⊗ q_{W<-F}
```

이 quaternion을 엘보우 축 `a_E^U`에 대한 twist와 나머지 swing으로 분해한다.

벡터부를 `v`라 할 때

```text
v_twist = (v·a_E^U) a_E^U
q_twist = normalize([w, v_twist])
θ_twist = 2 atan2( v_twist_norm, w )
```

이 각을 엘보우 후보각으로 쓴다.

이 방식의 장점:

- quaternion 기반이라 구현이 깔끔함
- relative orientation이 안정적이면 매우 직접적

단점:

- 6축 IMU의 yaw drift 차이에 더 민감
- 잘못된 축 캘리브레이션에 더 민감

### 6. 상완 2개 IMU를 의미 있게 쓰는 방법

상완 2개 IMU는 단순 중복이 아니라 다음 용도로 반드시 활용한다.

1. **강체 일관성 검사**
   - 두 센서가 같은 링크를 보므로 상대 오차가 작아야 함
2. **자세 평균**
   - 한 센서의 순간 오염을 다른 센서가 완화
3. **동적 구간 신뢰도 판정**
   - 둘의 quaternion 차이가 커지면 accel correction이 오염된 것일 가능성이 큼
4. **선택적 각가속도/강체 검증**
   - 위치가 다르므로 동적 가속도 차를 이용해 rigid-body consistency check 가능

### 7. 어깨 쪽 자세는 어디까지 가능한가

현재 구성에서 shoulder-chain 쪽은 다음 정도가 현실적이다.

- 상완의 **수직 대비 기울기**
  - reliable
- 기준 운동 평면 안에서의 상완 pitch 유사량
  - short-term usable
- 절대 shoulder yaw
  - not reliable

따라서 이번 모듈의 primary output은 다음으로 두는 것이 맞다.

- `upper_arm_tilt_deg`
- `forearm_tilt_deg`
- `elbow_angle_deg`

---

## 권장 최종 방법

### 1. 최종 추천 추정기

**주 방법**

- 각 IMU의 필터 quaternion 사용
- 기준 자세에서 센서-링크 고정 회전 오프셋 보정
- 상완 2개 IMU 평균으로 상완 링크 자세 생성
- 전완 IMU 1개로 전완 링크 자세 생성
- 힌지축 기반 평면 사영 signed angle로 엘보우 각 계산

**보조 방법**

- 상대 quaternion의 swing-twist 분해로 엘보우 후보각 계산
- 주 방법과 차이가 과도하면 품질 경고

### 2. 왜 이 구성을 메인으로 추천하는가

- 6축 IMU에서 가장 취약한 것은 absolute heading
- 하지만 우리가 진짜 필요한 것은 **엘보우 상대각**
- 링크 방향 벡터 기반 평면 사영법은 힌지 구조를 직접 사용하므로 가장 robust
- 상완 2개 IMU를 그대로 품질 향상에 쓸 수 있음
- 전완 top offset은 accelerometer gain 조절과 confidence 판정으로 다루기 쉬움

### 3. 권장 파이프라인

1. raw IMU 읽기
2. 각 IMU별 기존 필터로 quaternion 생성
3. 기준 90도 자세 static calibration
4. 센서-링크 회전 오프셋 적용
5. 상완 2개 IMU 자세 평균
6. 전완 자세 추정
7. 링크 방향 벡터 계산
8. 힌지축 기반 엘보우 각 계산
9. backup twist angle 계산
10. 두 결과 비교 및 sanity check
11. 결과 출력

---

## 코드 구조

이 폴더에는 Arduino Uno에 바로 붙일 수 있는 뼈대를 넣었다.

```text
imu각추정/
├─ README.md
├─ Estimator.h
├─ Estimator.cpp
└─ demo.ino
```

### 1. Arduino 측 역할

- IMU 3개에서 필터 결과 quaternion 수신 또는 생성
- reference calibration 수행
- optional elbow-axis calibration 수행
- 링크 자세와 엘보우 각 계산
- serial 출력

### 2. Host 측으로 넘겨도 좋은 역할

Uno 자원이 빠듯하면 다음은 host로 넘겨도 된다.

- quaternion 필터 자체
- elbow-axis calibration의 고유벡터 계산
- 장시간 로깅과 drift 분석
- 시각화

### 3. 핵심 자료구조

- `ImuPoseSample`
  - quaternion
  - accel
  - gyro
- `ImuMountModel`
  - 센서 위치
  - 기준 quaternion
  - 센서-링크 회전 오프셋
- `JointAngleEstimate`
  - 상완/전완 방향
  - 주/보조 엘보우 각
  - 신뢰도 및 일관성 지표
- `JointAngleEstimator`
  - calibration
  - update
  - sanity check

### 4. 코드에서 구현된 핵심 함수

- `beginReferenceCalibration()`
- `pushReferenceCalibrationSample()`
- `finishReferenceCalibration()`
- `beginElbowAxisCalibration()`
- `pushElbowAxisCalibrationSample()`
- `finishElbowAxisCalibration()`
- `update()`
- `estimateElbowAnglePrimaryRad()`
- `estimateElbowAngleBackupRad()`
- `estimateRotationalSpecificForceMps2()`

---

## 단계별 구현 계획

### Phase 1. 각 IMU 자세가 정상인지 확인

목표:

- 3개 센서의 quaternion/roll-pitch가 정상 출력되는지 확인
- 정지 시 drift와 노이즈 크기 확인

체크:

- 정지 상태에서 roll/pitch 흔들림이 작은지
- 기준 자세에서 U1/U2/F1의 orientation이 반복 가능하게 나오는지

### Phase 2. 상완 2개 IMU 캘리브레이션과 융합

목표:

- 기준 자세에서 U1/U2 회전 오프셋 저장
- 상완 링크 quaternion 2개를 평균해 하나의 `q_WU` 생성

체크:

- 정적 기준 자세에서 두 센서 링크 복원 결과가 거의 같은지
- `upper_link_consistency_deg`가 충분히 작은지

### Phase 3. 전완 링크 자세 추정

목표:

- F1 기준 자세 오프셋 저장
- `q_WF` 복원
- 전완 IMU top offset 정보를 모델에 반영

체크:

- 천천히 움직일 때 전완 기울기 추정이 기대와 비슷한지
- 빠른 움직임에서 accel 기반 correction이 과도하게 흔들리지 않는지

### Phase 4. 엘보우 상대각 추정

목표:

- 주 방법: 링크 방향 벡터 기반 signed angle
- 보조 방법: relative quaternion twist angle

체크:

- 기준 자세에서 0도 근처로 나오는지
- 완전히 편 자세, 더 굽힌 자세에서 단조롭게 변하는지

### Phase 5. 수동 자세 시험과 drift 평가

목표:

- 반복성, drift, 동적 오차 확인
- U1/U2와 F1의 장기 일관성 확인

---

## 검증 계획

### 1. 기준 자세 시험

조건:

- 기준 90도 자세로 10회 반복

확인:

- `elbow_angle_deg` 평균이 0도 근처인지
- 반복 표준편차가 충분히 작은지
- 상완/전완 tilt가 기준값과 일치하는지

### 2. 알려진 각도 시험

방법:

- 각도기 또는 간단한 지그로 엘보우를 `0°, 15°, 30°, 45°, 60°, 75°, 90°` 상대 이동

확인:

- 각 구간의 평균 오차
- 히스테리시스
- 증가/감소 시 차이

### 3. 반복 운동 시험

방법:

- 기준 자세 ↔ 더 굽힌 자세를 20회 반복

확인:

- cycle-to-cycle repeatability
- 상완 2개 IMU 일관성
- peak motion에서의 추정 튐

### 4. drift 시험

방법:

- 한 자세에서 60초~180초 정지

확인:

- 주 방법 엘보우 각 drift
- 보조 방법 drift
- 두 값의 차이 증가 여부

### 5. 전완 top-mount 영향 시험

방법:

- 천천히 움직인 로그와 빠르게 움직인 로그를 비교

확인:

- `|a|-g` 편차가 클 때 오차가 커지는지
- `ω`가 클 때 전완 결과가 더 흔들리는지
- top offset 보정 또는 gain scheduling 전후 비교

### 6. 상완 듀얼 IMU 일관성 시험

방법:

- 상완만 움직이거나 전체를 천천히 흔들기

확인:

- `upper_link_consistency_deg`
- U1/U2 개별 링크 방향 벡터 차이
- 어떤 센서가 더 자주 튀는지

---

## 간단 권장안 요약

### 최종 추천 추정기

- **주 방법**
  - 기준 자세 캘리브레이션
  - 상완 2개 IMU 링크 자세 평균
  - 전완 1개 IMU 링크 자세 계산
  - 힌지축 기반 평면 사영 signed angle로 엘보우 각 계산

- **보조 방법**
  - 상대 quaternion swing-twist 분해
  - 주 방법과 비교하여 품질 감시

### 구현 우선순위

1. 기준 자세 캘리브레이션
2. 상완 2개 IMU 융합
3. 링크 방향 벡터 계산
4. 엘보우 주 각도 계산
5. 보조 각도 계산
6. top offset 기반 신뢰도 조절

### 핵심 pseudocode

```text
read filtered quaternion/gyro/accel from U1, U2, F1

if reference calibration mode:
    accumulate quaternion averages
    accumulate gyro bias
    return

q_WU1 = q_WS_U1 * conj(q_ref_U1)
q_WU2 = q_WS_U2 * conj(q_ref_U2)
q_WF  = q_WS_F1 * conj(q_ref_F1)

q_WU = weighted_average(q_WU1, q_WU2)

u_W = rotate(q_WU, z_upper_ref)
f_W = rotate(q_WF, x_forearm_ref)
a_W = rotate(q_WU, elbow_axis_upper_ref)

u_p = normalize(u_W - dot(u_W, a_W) * a_W)
f_p = normalize(f_W - dot(f_W, a_W) * a_W)

theta_primary = atan2(dot(a_W, cross(u_p, f_p)), dot(u_p, f_p)) - theta_ref

q_UF = conj(q_WU) * q_WF
theta_backup = swing_twist_about_axis(q_UF, elbow_axis_upper_ref) - theta_backup_ref

report:
    upper_arm_tilt
    forearm_tilt
    elbow_primary
    elbow_backup
    upper_link_consistency
```

---

## 실무 메모

- MPU-6050은 주소가 2개뿐이라 Uno에 3개를 붙이려면 일반적으로 `TCA9548A` 같은 I2C 멀티플렉서가 필요하다.
- 전완 IMU의 **상단 오프셋 높이 `h_top`은 꼭 실측**해야 한다.
- 상완 2개 IMU는 반드시 둘 다 써야 한다. 하나는 측정, 다른 하나는 품질 검증 겸 평균용이다.
- 장기 yaw drift는 피할 수 없으므로 결과는 **엘보우 상대각 중심으로 해석**하는 것이 맞다.
- 빠른 동작에서 accelerometer correction을 세게 두면 오히려 전완 추정이 망가질 수 있다.
