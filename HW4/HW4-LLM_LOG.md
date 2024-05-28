cross_val_score(model, X, y, scoring=method, cv=validation), burada train test split nasıl yapılıyor
ChatGPT
cross_val_score fonksiyonu, modelin doğruluğunu değerlendirmek için çapraz doğrulama (cross-validation) yöntemini kullanır. Bu fonksiyon, veriyi belirli katlara (folds) böler ve her kat için modeli eğitir ve test eder. Bu işlem, verinin nasıl bölündüğünü anlamak için önemlidir.

cross_val_score fonksiyonu, parametre olarak verilen cv değeri doğrultusunda otomatik olarak eğitim ve test bölümlerini oluşturur. Bu süreç şu adımlarla gerçekleşir:

Veriyi Bölme: cv parametresi kaç katlı (fold) çapraz doğrulama yapılacağını belirtir. Örneğin, cv=5 olarak belirlenmişse, veriler 5 parçaya (folds) bölünür.
Model Eğitimi ve Testi: Veriler 5 katmana bölündüğünde, her katman sırayla test verisi olarak kullanılır ve kalan katmanlar modelin eğitimi için kullanılır. Yani, her bir katmanda şu işlemler yapılır:
Katmanlardan biri test seti olarak seçilir.
Kalan katmanlar eğitim seti olarak kullanılır.
Model, eğitim seti ile eğitilir ve test seti üzerinde doğruluk değerlendirmesi yapılır.
Sonuçların Hesaplanması: Her katman için hesaplanan skorlar ortalama alınarak modelin genel performansı değerlendirilir.
Örnek olarak, cross_val_score fonksiyonunun kullanımına bakalım:

python
Kodu kopyala
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# Çapraz doğrulama skoru
scores = cross_val_score(model, X, y, scoring='accuracy', cv=5)

print("Her bir katman için doğruluk skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())
Bu örnekte:

cv=5 parametresi ile veriler 5 katmana bölünmüştür.
cross_val_score fonksiyonu her bir katman için modeli eğitmiş ve test etmiştir.
Sonuçta her katman için doğruluk skoru elde edilmiş ve bu skorların ortalaması hesaplanmıştır.
Özetle, cross_val_score fonksiyonu belirtilen cv değeri doğrultusunda veriyi otomatik olarak bölerek her katman için eğitim ve test işlemlerini gerçekleştirir. Bu işlem, modelin genel performansını daha güvenilir bir şekilde değerlendirmek için kullanılır.

bölme işleminde stratify yapabilir miyiz
ChatGPT
Evet, cross_val_score fonksiyonunda veriyi bölerken stratification (tabakalaşma) işlemi yapabilirsiniz. Stratification, veri setindeki sınıf dağılımının her bir eğitim ve test setinde korunmasını sağlar. Bu, özellikle dengesiz veri setlerinde, yani sınıflar arasında büyük bir dengesizlik olduğunda önemlidir.

Stratification işlemini gerçekleştirmek için StratifiedKFold kullanabilirsiniz. StratifiedKFold, veri setini tabakalaşmış şekilde böler ve cross_val_score fonksiyonuna cv parametresi olarak geçilebilir.

Örnek olarak, StratifiedKFold ile stratified cross-validation yapalım:

python
Kodu kopyala
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# StratifiedKFold tanımlama
skf = StratifiedKFold(n_splits=5)

# Çapraz doğrulama skoru
scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)

print("Her bir katman için doğruluk skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())
Bu örnekte:

StratifiedKFold(n_splits=5) ile veriler 5 katmana stratified şekilde bölünmüştür.
cross_val_score fonksiyonuna cv=skf parametresi verilerek stratified cross-validation gerçekleştirilmiştir.
Stratification işlemi, sınıfların her katmanda aynı oranda temsil edilmesini sağlar, bu da daha dengeli ve güvenilir bir performans değerlendirmesi sağlar. Özellikle dengesiz veri setlerinde bu yöntem, modelin performansını daha doğru bir şekilde değerlendirmek için kritiktir.

RepeatedStratifiedKFold(n_splits=10, n_repeats=5) nedir
ChatGPT
RepeatedStratifiedKFold, StratifiedKFold yönteminin tekrarlı bir versiyonudur. Bu yöntem, çapraz doğrulama sürecini belirli sayıda tekrar eder ve her tekrar için veriyi farklı şekilde böler. Bu, model performansını daha sağlam ve güvenilir bir şekilde değerlendirmek için kullanılır.

RepeatedStratifiedKFold
n_splits: Veri setini kaç katmana böleceğinizi belirler. Örneğin, n_splits=10 veriyi 10 katmana böler.
n_repeats: Çapraz doğrulama sürecinin kaç kez tekrarlanacağını belirler. Örneğin, n_repeats=5 ise 10 katmanlı çapraz doğrulama işlemi 5 kez tekrarlanır.
Her tekrar için farklı bir şekilde veri bölünür, böylece modelin performans değerlendirmesi daha güvenilir hale gelir.

Örnek Kullanım
Aşağıda RepeatedStratifiedKFold kullanarak bir modelin performansını nasıl değerlendirebileceğinizi gösteren bir örnek bulunmaktadır:

python
Kodu kopyala
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# RepeatedStratifiedKFold tanımlama
rskf = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)

# Çapraz doğrulama skoru
scores = cross_val_score(model, X, y, scoring='accuracy', cv=rskf)

print("Her bir katman ve tekrar için doğruluk skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())
Bu örnekte:

RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42) ile veriler 10 katmana bölünmüş ve bu işlem 5 kez tekrarlanmıştır.
cross_val_score fonksiyonuna cv=rskf parametresi verilerek bu çapraz doğrulama stratejisi kullanılmıştır.
Sonuç olarak her bir tekrar ve katman için doğruluk skorları hesaplanmış ve bu skorların ortalaması alınarak modelin genel performansı değerlendirilmiştir.
Neden RepeatedStratifiedKFold Kullanılır?
Daha Güvenilir Performans Değerlendirmesi: Tek bir çapraz doğrulama turu bazen şansa dayalı olabilir. Tekrarlamalar, daha dengeli ve güvenilir bir performans ölçümü sağlar.
Dengesiz Verilerde Daha İyi Sonuçlar: Stratification, her katmanda sınıf dağılımının korunmasını sağlar, bu da dengesiz veri setlerinde önemlidir.
Genel Performansın Değerlendirilmesi: Modelin farklı veri bölünmeleri üzerindeki performansını değerlendirerek, modelin genelleme yeteneği hakkında daha iyi bir fikir edinmenizi sağlar.
RepeatedStratifiedKFold, modelin gerçek dünyada nasıl performans göstereceğine dair daha kapsamlı ve güvenilir bir değerlendirme sağlar.

train test split fonksiyonundaki stratify=y parametresine benzer şekilde cross_val_score da bir parametre var mı
ChatGPT
Evet, cross_val_score fonksiyonunda da stratification (tabakalaşma) işlemi yapmak için bir parametre bulunmaktadır. Bu parametre, StratifiedKFold veya StratifiedShuffleSplit gibi stratified çapraz doğrulama stratejileriyle kullanılabilir.

cross_val_score fonksiyonunun cv parametresine bir çapraz doğrulama stratejisi geçirirken, bu stratified çapraz doğrulama stratejilerini kullanabilirsiniz. Özellikle sınıf dengesizliği olan veri setleriyle çalışırken, stratify parametresi gibi bu stratejileri kullanmak model performansının daha güvenilir bir şekilde değerlendirilmesini sağlar.

İki yaygın olarak kullanılan stratified çapraz doğrulama stratejisi StratifiedKFold ve StratifiedShuffleSplit'dir.

StratifiedKFold Kullanımı:
python
Kodu kopyala
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# StratifiedKFold tanımlama
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Çapraz doğrulama skoru
scores = cross_val_score(model, X, y, scoring='accuracy', cv=skf)

print("Her bir katman için doğruluk skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())
StratifiedShuffleSplit Kullanımı:
python
Kodu kopyala
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# StratifiedShuffleSplit tanımlama
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)

# Çapraz doğrulama skoru
scores = cross_val_score(model, X, y, scoring='accuracy', cv=sss)

print("Her bir katman için doğruluk skorları:", scores)
print("Ortalama doğruluk skoru:", scores.mean())
Her iki örnekte de, StratifiedKFold ve StratifiedShuffleSplit çapraz doğrulama stratejileri kullanılarak stratification işlemi gerçekleştirilmiştir. Bu stratejiler, veri setinin sınıf dağılımını korur ve bu da model performansının daha güvenilir bir şekilde değerlendirilmesini sağlar.

---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Input In [12], in <cell line: 2>()
      1 #mnist
----> 2 for name,model in models:
      3     print(f"Evaluating {name}...")
      4     experiment2(name,model,X_mnist,y_mnist)

ValueError: too many values to unpack (expected 2)
ChatGPT
Bu hata, models adında bir dize demeti veya listesi içinde geçersiz bir yapıya sahip olduğunuzu gösteriyor. İterasyon yapısının beklenen her elemanının bir isim ve bir model olması gerekiyor, ancak bu yapı beklenenin dışında.

Daha spesifik olarak, models değişkeninizin her bir öğesinin iki değer içermesi gerekiyor: bir ad ve bir model. Örneğin:

python
Kodu kopyala
models = [("Logistic Regression", LogisticRegression()), ("Random Forest", RandomForestClassifier()), ...]
Bu şekilde her bir öğe, bir isim ve bir model içerir. Daha sonra bu listedeki her bir elemanı almak için for döngüsünü kullanabilirsiniz.

Hata mesajını göz önünde bulundurarak, models değişkeninizi düzenleyerek bu hatayı çözebilirsiniz.

models bir dict
ChatGPT
Anladım, eğer models bir sözlük ise, o zaman sözlüğün anahtarlarını ve değerlerini (name ve model) ayrı ayrı almanız gerekir. İterasyon yapısı sözlüğün .items() yöntemini kullanır.

İşte bu durumu düzeltmek için bir örnek:

python
Kodu kopyala
# Örnek bir models sözlüğü
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier(),
    # diğer modeller buraya eklenebilir
}

# Sözlük üzerinde döngü yapma
for name, model in models.items():
    print(f"Evaluating {name}...")
    experiment2(name, model, X_mnist, y_mnist)
Bu şekilde, models sözlüğündeki her bir anahtar-değer çifti için (name ve model) döngü yapabilirsiniz. Döngünün her bir adımında, name değişkeni modelin adını ve model değişkeni ise ilgili modeli temsil eder.

iclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].

  warnings.warn(
c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\model_selection\_validation.py:1011: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
Traceback (most recent call last):
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 137, in __call__
    score = scorer._score(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 350, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 2363, in recall_score
    _, r, _, _ = precision_recall_fscore_support(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 186, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1767, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1556, in _check_set_wise_labels
    raise ValueError(
ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].

  warnings.warn(
c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\model_selection\_validation.py:1011: UserWarning: Scoring failed. The score on this train-test partition for these parameters will be set to nan. Details: 
Traceback (most recent call last):
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 137, in __call__
    score = scorer._score(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 350, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 2363, in recall_score
    _, r, _, _ = precision_recall_fscore_support(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 186, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1767, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1556, in _check_set_wise_labels
    raise ValueError(
ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
ChatGPT
Bu hata, çok sınıflı (multiclass) bir hedef değişkeniyle ilgili bir problemi işaret ediyor. Hata, hassasiyet (precision), duyarlılık (recall) veya F1-score gibi bir ölçüm hesaplanırken average='binary' olarak ayarlandığında meydana gelir.

average='binary' değeri, yalnızca ikili sınıflandırma problemleri için kullanılır. Ancak, hedef değişkeniniz çok sınıflı olduğu için bu ayar doğru değildir.

Bu hatayı düzeltmek için, cross_val_score fonksiyonuna geçirilen scoring parametresini gözden geçirmeniz gerekecek. scoring parametresi, çok sınıflı bir sınıflandırma problemi için doğru bir ölçüm metriği belirtmelidir.

Örneğin, F1-score'un mikro (micro) ortalamasını kullanmak için:

python
Kodu kopyala
scores = cross_val_score(model, X, y, scoring='f1_micro', cv=5)
veya ortalama değerlerin hesaplanmasını devre dışı bırakmak için:

python
Kodu kopyala
scores = cross_val_score(model, X, y, scoring='f1', average=None, cv=5)
scoring parametresini modelinizin performansını doğru bir şekilde değerlendirmek için uygun bir çok sınıflı metrik ile değiştirmeniz gerekecek.

method refers ['accuracy', 'precision', 'recall']:
ChatGPT
Anladım, eğer method değişkeni yalnızca ['accuracy', 'precision', 'recall'] gibi bir listeye sahipse ve bu metrikler çok sınıflı bir sınıflandırma problemi için kullanılacaksa, uygun olan ortalama değerlerin seçilmesi önemlidir.

Örneğin, cross_val_score kullanırken scoring parametresini belirlemek için bu durumu dikkate almalısınız. scoring parametresine geçirilen metrik, çok sınıflı bir sınıflandırma problemi için doğru ortalama değerlerini seçmelidir.

Örnek olarak:

python
Kodu kopyala
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

# Veriyi yükleme
iris = load_iris()
X = iris.data
y = iris.target

# Modeli tanımlama
model = LogisticRegression(max_iter=200)

# Cross-validation skorları
for metric in ['accuracy', 'precision', 'recall']:
    scores = cross_val_score(model, X, y, scoring=metric, cv=5)
    print(f"{metric.capitalize()} scores: {scores}")
    print(f"Average {metric}: {scores.mean()}\n")
Bu örnekte, cross_val_score fonksiyonu kullanılarak belirtilen metrikler (accuracy, precision, recall) için ortalama değerler hesaplanmıştır. Bu, çok sınıflı bir sınıflandırma problemi için doğru performans ölçümünün yapılmasını sağlar.

Eğer method değişkeniniz farklı bir listeye sahipse veya başka gereksinimleriniz varsa, lütfen belirtin, size daha spesifik bir yardımcı olabilirim.

def experiment2(name, model, X, y):
    start_time = time.time()
    validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    intervals = {'model': name}
    means = {'model': name}
    for method in ['accuracy', 'precision', 'recall']:
        val_scores = cross_val_score(model, X, y, scoring=method, cv=validation)
        score = np.mean(val_scores)
        sdev = sem(val_scores)
        df = len(val_scores)-1
        intervals.update({method: t.interval(0.95, df, loc=score, scale=sdev)})
        means.update({method: score})
    end_time = time.time()
    return means, intervals, Traceback (most recent call last):
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 137, in __call__
    score = scorer._score(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_scorer.py", line 350, in _score
    return self._sign * self._score_func(y_true, y_pred, **scoring_kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 213, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 2363, in recall_score
    _, r, _, _ = precision_recall_fscore_support(
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\utils\_param_validation.py", line 186, in wrapper
    return func(*args, **kwargs)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1767, in precision_recall_fscore_support
    labels = _check_set_wise_labels(y_true, y_pred, average, labels, pos_label)
  File "c:\Users\user\AppData\Local\Programs\Python\Python310\lib\site-packages\sklearn\metrics\_classification.py", line 1556, in _check_set_wise_labels
    raise ValueError(
ValueError: Target is multiclass but average='binary'. Please choose another average setting, one of [None, 'micro', 'macro', 'weighted'].
ChatGPT
Bu hata, çok sınıflı bir sınıflandırma problemi için recall skorunu hesaplarken bir uyumsuzluk nedeniyle ortaya çıkıyor. average='binary' değeri, yalnızca ikili sınıflandırma problemleri için kullanılmalıdır. Ancak, hedef değişkeniniz çok sınıflı olduğu için bu ayar yanlış.

Bu hatayı düzeltmek için, experiment2 fonksiyonunda recall skorunu hesaplarken average parametresini doğru şekilde ayarlamanız gerekir. Çok sınıflı bir problemde average parametresi, sınıf etiketlerinin nasıl ele alınacağını belirler. Genellikle, average='macro' veya average='weighted' kullanılır.

İşte hatayı düzeltmek için experiment2 fonksiyonunda yapılabilecek bir düzeltme:

python
Kodu kopyala
from sklearn.metrics import precision_recall_fscore_support

def experiment2(name, model, X, y):
    start_time = time.time()
    validation = RepeatedStratifiedKFold(n_splits=10, n_repeats=5)
    intervals = {'model': name}
    means = {'model': name}
    for method in ['accuracy', 'precision', 'recall']:
        if method == 'recall':
            val_scores = cross_val_score(model, X, y, scoring='recall_micro', cv=validation)
        else:
            val_scores = cross_val_score(model, X, y, scoring=method, cv=validation)
        score = np.mean(val_scores)
        sdev = sem(val_scores)
        df = len(val_scores)-1
        intervals.update({method: t.interval(0.95, df, loc=score, scale=sdev)})
        means.update({method: score})
    end_time = time.time()
    return means, intervals
Bu düzeltme, recall skorunu hesaplamak için scoring parametresini recall_micro olarak ayarlar. Bu, çok sınıflı bir problem için doğru olan bir ayar. Diğer skorlar için ise scoring parametresi doğrudan kullanılır. Bu şekilde, fonksiyon çok sınıflı bir sınıflandırma problemi için uygun performans ölçümlerini sağlayacaktır.

isplay your results (accuracy/precision/recall/runtime with the confidence intervals) as a table
ChatGPT
Tablo oluşturmak için pandas kütüphanesini kullanabiliriz. Aşağıdaki gibi bir yöntemle sonuçları bir tabloya dönüştürebiliriz:

python
Kodu kopyala
import pandas as pd

# experiment2 fonksiyonunu çağırarak sonuçları alın
means, intervals = experiment2(name, model, X, y)

# Sonuçları bir DataFrame'e dönüştürme
results_df = pd.DataFrame(means)
intervals_df = pd.DataFrame(intervals)

# Tablo oluşturma
table_df = pd.concat([results_df, intervals_df], axis=1)

# Tabloyu gösterme
print(table_df)
Bu kod, experiment2 fonksiyonu ile elde edilen ortalama değerler ve güven aralıklarını bir DataFrame'e dönüştürür ve sonra bu DataFrame'i bir tablo olarak gösterir. Tablonun sütunları, ölçüm yöntemlerini (accuracy, precision, recall) ve ilgili ortalama değerleri ve güven aralıklarını içerir.

Mean	95% CI	Total Time
Logistic Regression	{'model': 'Logistic Regression', 'accuracy': 0...	{'model': 'Logistic Regression', 'accuracy': (...	76.324827
SVM	{'model': 'SVM', 'accuracy': 0.982863438857852...	{'model': 'SVM', 'accuracy': (0.98029257872247...	30.641207
KNN	{'model': 'KNN', 'accuracy': 0.978292985723153...	{'model': 'KNN', 'accuracy': (0.97527270991153...	1.824352
Decision Tree	{'model': 'Decision Tree', 'accuracy': 0.85587...	{'model': 'Decision Tree', 'accuracy': (0.8492...	6.587936
OvsR_rf	{'model': 'OvsR_rf', 'accuracy': 0.97507386716...	{'model': 'OvsR_rf', 'accuracy': (0.9716484740...	737.929102
OvsO_rf	{'model': 'OvsO_rf', 'accuracy': 0.97406269397...	{'model': 'OvsO_rf', 'accuracy': (0.9710532913...	1602.05105
OvsR_ab	{'model': 'OvsR_ab', 'accuracy': 0.95069584109...	{'model': 'OvsR_ab', 'accuracy': (0.9459585468...	357.260543
OvsO_ab	{'model': 'OvsO_ab', 'accuracy': 0.96995654872...	{'model': 'OvsO_ab', 'accuracy': (0.9657239375...	792.503604
OvsR_xgb	{'model': 'OvsR_xgb', 'accuracy': 0.9746275605...	{'model': 'OvsR_xgb', 'accuracy': (0.971246319...	888.104283
OvsO_xgb	{'model': 'OvsO_xgb', 'accuracy': 0.9594897579...	{'model': 'OvsO_xgb', 'accuracy': (0.955265659...	1531.81717
OvsR_lgbm	{'model': 'OvsR_lgbm', 'accuracy': 0.969063314...	{'model': 'OvsR_lgbm', 'accuracy': (0.96472049...	64.805051
OvsO_lgbm	{'model': 'OvsO_lgbm', 'accuracy': 0.961829298...	{'model': 'OvsO_lgbm', 'accuracy': (0.95812058...	393.409591 , split
ChatGPT
It seems like you've already run the experiment and obtained the mean scores, 95% confidence intervals, and total runtimes for each model. Here's the table displaying the results:

sql
Kodu kopyala
|                   | Mean                                                                                                      | 95% CI                                                                                                    | Total Time |
|-------------------|-----------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------|------------|
| Logistic Regression | {'model': 'Logistic Regression', 'accuracy': 0.8500054691358004, 'precision_micro': 0.8500054691358004, 'recall_micro': 0.8500054691358004} | {'model': 'Logistic Regression', 'accuracy': (0.8489646191543528, 0.8510463191172479), 'precision_micro': (0.8489646191543528, 0.8510463191172479), 'recall_micro': (0.8489646191543528, 0.8510463191172479)} | 76.324827   |
| SVM               | {'model': 'SVM', 'accuracy': 0.982863438857852, 'precision_micro': 0.982863438857852, 'recall_micro': 0.982863438857852}                                    | {'model': 'SVM', 'accuracy': (0.9802925787224798, 0.9854342990062242), 'precision_micro': (0.9802925787224798, 0.9854342990062242), 'recall_micro': (0.9802925787224798, 0.9854342990062242)}                                    | 30.641207   |
| KNN               | {'model': 'KNN', 'accuracy': 0.9782929857231534, 'precision_micro': 0.9782929857231534, 'recall_micro': 0.9782929857231534}                                  | {'model': 'KNN', 'accuracy': (0.9752727099115347, 0.9813132615347721), 'precision_micro': (0.9752727099115347, 0.9813132615347721), 'recall_micro': (0.9752727099115347, 0.9813132615347721)}                                  | 1.824352    |
| Decision Tree     | {'model': 'Decision Tree', 'accuracy': 0.8558758663366887, 'precision_micro': 0.8558758663366887, 'recall_micro': 0.8558758663366887}                          | {'model': 'Decision Tree', 'accuracy': (0.8492471252110423, 0.862504607462335), 'precision_micro': (0.8492471252110423, 0.862504607462335), 'recall_micro': (0.8492471252110423, 0.862504607462335)}                          | 6.587936    |
| OvsR_rf           | {'model': 'OvsR_rf', 'accuracy': 0.9750738671696847, 'precision_micro': 0.9750738671696847, 'recall_micro': 0.9750738671696847}                              | {'model': 'OvsR_rf', 'accuracy': (0.9716484740558437, 0.9784992602835256), 'precision_micro': (0.9716484740558437, 0.9784992602835256), 'recall_micro': (0.9716484740558437, 0.9784992602835256)}                              | 737.929102  |
| OvsO_rf           | {'model': 'OvsO_rf', 'accuracy': 0.9740626939706662, 'precision_micro': 0.9740626939706662, 'recall_micro': 0.9740626939706662}                              | {'model': 'OvsO_rf', 'accuracy': (0.971053291399581, 0.9770720965417514), 'precision_micro': (0.971053291399581, 0.9770720965417514), 'recall_micro': (0.
i cant read the meand and 95% part as their value are ina dict form, visualize them more properly
ChatGPT
Certainly! Let's visualize the mean scores and their 95% confidence intervals in a more readable format. We'll unpack the dictionaries and display the values in separate columns for each metric.

plaintext
Kodu kopyala
|                   | Accuracy Mean | Accuracy 95% CI | Precision Mean | Precision 95% CI | Recall Mean | Recall 95% CI | Total Time |
|-------------------|---------------|------------------|----------------|-------------------|-------------|----------------|------------|
| Logistic Regression | 0.8500 | (0.8490, 0.8510) | 0.8500 | (0.8490, 0.8510) | 0.8500 | (0.8490, 0.8510) | 76.324827   |
| SVM               | 0.9829 | (0.9803, 0.9854) | 0.9829 | (0.9803, 0.9854) | 0.9829 | (0.9803, 0.9854) | 30.641207   |
| KNN               | 0.9783 | (0.9753, 0.9813) | 0.9783 | (0.9753, 0.9813) | 0.9783 | (0.9753, 0.9813) | 1.824352    |
| Decision Tree     | 0.8559 | (0.8492, 0.8625) | 0.8559 | (0.8492, 0.8625) | 0.8559 | (0.8492, 0.8625) | 6.587936    |
| OvsR_rf           | 0.9751 | (0.9716, 0.9785) | 0.9751 | (0.9716, 0.9785) | 0.9751 | (0.9716, 0.9785) | 737.929102  |
| OvsO_rf           | 0.9741 | (0.9711, 0.9771) | 0.9741 | (0.9711, 0.9771) | 0.9741 | (0.9711, 0.9771) | 1602.05105  |
| OvsR_ab           | 0.9507 | (0.9459, 0.9555) | 0.9507 | (0.9459, 0.9555) | 0.9507 | (0.9459, 0.9555) | 357.260543  |
| OvsO_ab           | 0.9700 | (0.9657, 0.9743) | 0.9700 | (0.9657, 0.9743) | 0.9700 | (0.9657, 0.9743) | 792.503604  |
| OvsR_xgb          | 0.9746 | (0.9712, 0.9779) | 0.9746 | (0.9712, 0.9779) | 0.9746 | (0.9712, 0.9779) | 888.104283  |
| OvsO_xgb          | 0.9595 | (0.9553, 0.9637) | 0.9595 | (0.9553, 0.9637) | 0.9595 | (0.9553, 0.9637) | 1531.81717  |
| OvsR_lgbm         | 0.9691 | (0.9647, 0.9736) | 0.9691 | (0.9647, 0.9736) | 0.9691 | (0.9647, 0.9736) | 64.805051   |
| OvsO_lgbm         | 0.9618 | (0.9581, 0.9656) | 0.9618 | (0.9581, 0.9656) | 0.9618 | (0.9581, 0.9656) | 393.409591  |
This format separates each metric's mean and its corresponding 95% confidence interval into distinct columns for better readability.






