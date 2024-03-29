### BoostResNet

Репозиторий содержит простую PyTorch имплементацию статьи [Learning Deep ResNet Blocks Sequentially using Boosting Theory](https://arxiv.org/abs/1706.04964), в которой предлагается новый метод обучения глубоких resnet сетей BoostResNet, использующий теорию бустинга. Кратко, суть метода заключается в последовательном обучении residual блоков сети, что позволяет сильно экономить ресурсы. В качестве примера используется 50-слойная архитектура ResNet, на вход которой подаются нормализованные изображения 32 х 32, но которую можно легко адаптировать под вход любой размерности.    

BoostResNet имеет 2 основных применения:
* Сильный дефицит видеопамяти. В VRAM может находится только текущий обучаемый блок. (Не рекомендуется, так как много работы ляжет на CPU)
* Ускорение обучения. BoostResNet можно применить для первичной тренировки модели, которую затем можно дообучить, используя классический back propagation

Запуск      

```python boostresnet.py```

