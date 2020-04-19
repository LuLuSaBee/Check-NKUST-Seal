# In Micrsoft Custom Vision Training

## is_Seal

04/15/2020 11:30 Recall 46.7% Precision 90%

04/15/2020 13:10 Recall 32.0% Precision 100%

04/15/2020 15:02 Recall 52.0% Precision 100% (Use Advance)

## is_Hand

04/15/2020 12:53 Recall 40.0% Precision 70%


### 筆記

第一張會面進來後就判斷然後擷取圖片繼續盼下個

webcam一樣道理 第一個先接圖然後把他拿下去判斷 combian

GUI自己一個

找一個方法可以讓後面的form不被選到

接下來在__put_to_trial 截圖下來

### 測試時
自拍出現
[{'probability': 0.1698374, 'tagId': 0, 'tagName': 'is_hand', 'boundingBox': {'left': 0.51353402, 'top': 0.63164521, 'width': 0.43414026, 'height': 0.34110021}}, {'probability': 0.1206469, 'tagId': 1, 'tagName': 'not_hand', 'boundingBox': {'left': 0.25394067, 'top': 0.27637641, 'width': 0.39042807, 'height': 0.69788461}}]

但畫面中沒有手 因此判定是手就必須 0.2以上
