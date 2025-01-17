# mnist

自己實作 MLP 的過程，使用 mnist 的手寫集當作訓練資料

tensorflow 有官方程式碼: [mnist 訓練範例](https://www.tensorflow.org/datasets/keras_example)

但我自己想要深入知道 MLP 的內部過程，所以只使用了 numpy 作為運算工具

### 環境

使用 Google Colab，使用其免費的 GPU 來訓練

### 過程

1. 讀入輸入資料 ( 28 * 28 的圖片像素資訊 )，攤平成 784 * 1 (輸入層)
2. 撰寫 front_propagation ( 計算現在參數預測的答案和正確答案的差距 ) 以及 back_propagation ( 計算參數應該如何變動 ) 的程式碼
3. 將所有資料進行 front_propagation 以及 back_propagation ( 隱藏層 )，全部資料做完叫作一次 **"訓練"**，多次重複 **"訓練"** 這個步驟 ( 因為只訓練一次參數並不精準 )
4. 訓練好模型後，即可預測未知的手寫圖片

### 結果

```bash
Accuracy: 0.8787
```
