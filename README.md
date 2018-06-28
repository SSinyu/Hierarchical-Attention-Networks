TV_LOG
======
----------


3.27일 개편 전/후로 채널 맞추기. 
------
![channel_matching](https://github.com/SSinyu/TVLOG/blob/master/img/channel_matching.JPG)
* 3.26까지 모든 기록(채널 번호1)을 채널 번호2로 변경.
  ex) S38 >> S40 변경
* 엑셀 파일에 없지만, 데이터에는 기록된 채널은 변경하지 않음. 
ex) S190 은 채널 번호1,2 목록에 없으며 그냥 S190으로 유지.
-------


학습을 위한 input 생성
------
### ver.1
![build_dataset_1](https://github.com/SSinyu/TVLOG/blob/master/img/build_dataset_11.jpg)
* 큰 리스트는 한 유저의 tv 시청 기록, 작은 리스트는 각각 1일의 시청 기록. (계층적인 모델을 위해)
* 3일씩 input으로, 4일차의 첫번째 시청 채널을 target으로 하여 학습.

### ver.2
![build_dataset_2](https://github.com/SSinyu/TVLOG/blob/master/img/build_dataset_2.jpg)
* input은 같지만, 4일차의 첫번째 시청 채널이 속하는 카테고리를 target으로 하여 학습.
----------


모델링
------
### architecture
![model_architecture](https://github.com/SSinyu/TVLOG/blob/master/img/model_architecture.jpg)
*  http://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
----------


결과
------
![accuracy](https://github.com/SSinyu/TVLOG/blob/master/img/accc.png)
[각 채널이 속한 약 20개의 카테고리를 target으로 했을 때의 학습] (채널을 target으로 한 모델링 기록이 지워져서 그림을 넣지 못했습니다..)
* 약 230가지 채널을 target으로 했을 때 보다 더 나은 정확도를 보이지만 크게 차이를 보이진 않는다. (약 5%정도 높다)
* 그래프를 봤을 때, epoch을 늘리면 더 좋아질 것 같다.
----------


진행중인 것
------
![new_model](https://github.com/SSinyu/TVLOG/blob/master/img/new_model.jpg)
![build_dataset_1]()
1. 모델의 입력으로 들어가는 각 채널의 시간대 정보(아침, 저녁 등) 또는 그 채널에 대한 정보(드라마, 뉴스 등의 카테고리)를 one-hot vector로 만들어 기존 100차원으로 임베딩된 채널과 concatenation하여 입력으로 사용.
2. 모델의 정확도 향상을 위해 Day-level에서 softmax를 추가해 loss를 계산 후 gradient를 한번 더 역전파.





