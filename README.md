# twitter_disaster
교과융합프로젝트 - 국가재난정보와 트위터 기반 머신러닝 분석

# 연구의 결과
* 뉴스를 통해서 위험을 감지하고 예방 혹은 예측할 수 있는 가능성을 얻음.
1. 인공지능 모형의 입력값의 필요성
-인공지능 모형은 추론 가능한 가설을 만들어야 함
-가설: 뉴스를 통해서 위험을 감지하고 예방 혹은 예측할 수 있다
-뉴스의 작성 시점과 같은 정확한 시간 정보가 필요함
* 트위터처럼 즉시성을 갖는 뉴스를 통해 위험을 감지하는 것이 더 효과적임.
2. 예측 모형의 신뢰도 vs 신속성
-뉴스는 체계적으로 보도하기 위한 절차와 시간이 필요하기 때문에 신속성 떨어짐
-트위터는 신뢰도는 낮으나 재난 발생 시점에 가장 근접한 소식을 전달할 수 있다고 여김
-재난 상황에서는 신뢰도보다 신속성에 초점을 두어 대응하는 것이 중요
-신뢰도를 보완하기 위해서 과거 트위터 작성 시점에 재난발생 기록을 연동하여 재난 발생시 나타나는 트위터 글의 의미와 감성 패턴을 모형으로 구현
* 뉴스의 의미와 감성을 분석하는 방법으로 KoBERT(pre-trained된 학습결과를 사용하여 조금만 바꿔주는 fine-tuning 방식)를 활용.
3. 효율적인 의미와 감성 분석 방법
-BERT:사전 훈련된 모형(ex: 위키, 백과사전...)을 조금만 튜닝해서 다른 용도로 사용하는 방법의 효율성을 얻음
-
* 인공지능 모형을 만들기 위해서는 "국가재난포털"의 재난유형 정보와 같은 정확한 "사실 데이터"가 필요함.
1. 사실 데이터 기반의 예측&예방