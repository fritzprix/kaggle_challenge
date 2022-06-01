
# [Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

> Image Scene Classification of Multiclass

## Classes

- buildings
- forest
- glacier
- mountain
- sea
- street

## Note

- Training 중 Validation Loss의 fluctuation이 존재하는데 Learning Rate, Batch Size 등 Hyperparameter 설정을 바꿔도 현재까지는 별다른 효과는 없음. 단, training이 진행될 수록 이러한 Fluctuation이 점차 감소되는 경향이 있음.
- Validation Set 결과와 Test Set의 결과 차이가 큼. 첫 10샘플에 대해 (5/10)의 정확도 수준.
- grey scale 데이터가 존재하며 주로 glacier로 인식되는 경향이 있음 (색상 특징이 유사)
- grey scale로 전처리 후 학습 시키면 색상에 무관한 feature를 학습할 수 있을 것이라 예상됨?

