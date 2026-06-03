# StockSense AI: Tam Şeffaf ve Teknik Mülakat Rehberi 🚀

Bu doküman, StockSense AI projesinin "arka planında gerçekten ne çalıştığını", hangi kod dosyasında hangi mimarinin kurgulandığını ve tam (end-to-end) veri boru hattını (pipeline) uydurma yapmadan dürüstçe açıklamak için hazırlanmıştır. Mülakatta kod bazlı sorular geldiğinde direkt bu referansları kullanabilirsin.

---

## 1. Uçtan Uca (End-to-End) Veri ve İşleyiş Pipeline'ı

Bu projede veri nereden akar, nerede işlenir ve nihai RAG çıktısına nasıl dönüşür?

1. **Eğitim & İnce Ayar (Training & Fine-Tuning):**
   - **Quantitative Model (Sayısal):** `src/models/` altındaki Custom Transformer (veya Bi-LSTM), finansal OHLCV (fiyat/hacim) verilerini alır. Veri `src/data/` altındaki preprocessing adımlarından (Standard Scaling) geçerek modele girer. Model PyTorch üzerinde eğitilir.
   - **LLM Alignment (QLoRA):** Eğer LLM'i özel finansal jargona hizalamak istersek, `scripts/finetune_llm.py` çalışır. Qwen 2.5 modeli 4-bit (nf4) kuantize edilerek yüklenir ve LoRA adaptörleri (r=16, alpha=32) ile sadece attention katmanları eğitilir.

2. **Çıkarım ve Optimizasyon (Inference):**
   - Eğitilen PyTorch (Transformer/LSTM) modelleri, FastAPI üzerinden canlı (real-time) hizmet verebilmek için çok ağırdır. 
   - `make optimize` komutu ile modeller **Post-Training Dynamic Quantization (INT8)** işleminden geçer ve **ONNX Runtime** formatına derlenir (`test.onnx`). Bu sayede milisaniyenin altında tahmin (inference) hızlarına ulaşılır.

3. **RAG İşleyişi ve Bilgi Sentezi:**
   - **Adım A (Sayısal Tahmin):** ONNX modeli önümüzdeki X gün için bir fiyat tahmini (`predictions`) üretir.
   - **Adım B (Haber Duyarlılığı - NLP):** Aynı anda hisse ile ilgili haberler çekilir. `src/nlp/sentiment_analyzer.py` içerisindeki **FinBERT** modeli haberleri analiz edip `-1 ile 1` arası sayısal bir skor üretir.
   - **Adım C (Varlık Çıkarımı - NER):** `src/nlp/entity_extractor.py` dosyası çalışır. `dslim/bert-base-NER` modeli ve özel Regex kuralları ile haberlerde geçen şirket isimleri, ticker'lar (örn. AAPL) ve parasal değerler ($94.8B) çıkarılır.
   - **Adım D (RAG Prompting):** `src/nlp/llm_rag_chain.py` devreye girer. LangChain üzerinden (LCEL pipeline) bir sistem promptu kurgulanır. Nicel fiyat tahmini, FinBERT'in duyarlılık skoru ve ham haber metinleri LLM'e (Qwen 2.5) "Augmented Context" (Zenginleştirilmiş Bağlam) olarak verilir.
   - **Nihai Çıktı:** Qwen 2.5, LangChain (`ChatPromptTemplate | ChatOllama | StrOutputParser`) üzerinden sadece bu verileri baz alarak (halüsinasyon olmadan) Wall Street standartlarında profesyonel bir yatırım stratejisi metni (Markdown raporu) oluşturur.

---

## 2. "Kodda Nerede ve Neden?" - Mimari Tercihlerin Savunması

### ⚙️ Derin Öğrenme: Transformer Encoder (OHLCV + MHA + PE)
- **Kod Nerede?** `src/models/transformer_model.py` içerisinde `TransformerPredictor` sınıfında.
- **Nasıl Çalışır?** Model, `OHLCV` tabanlı zaman serisi (time-series) verilerini girdi olarak alır.
  1. **Positional Encoding (PE):** Transformer'lar sıralamayı (sequential order) doğal olarak bilmediği için, `sin(pos / 10000^(2i/d_model))` formülüne dayanan *Sinusoidal Positional Encoding* eklenir. Bu, modelin farklı uzunluklardaki zaman serilerine uyum sağlamasına olanak tanır.
  2. **Multi-Head Attention (MHA):** Model `nn.TransformerEncoderLayer` kullanır. RNN'lerin aksine geçmişe sırayla gitmek yerine, tüm tarihsel günlere (global dependencies) aynı anda odaklanarak volatil piyasa verilerini analiz eder.
  3. Eğitim kararlılığı için `norm_first=True` (Pre-norm) kullanılır ve son adımda `encoded.mean(dim=1)` ile *Global Average Pooling* yapılarak tahminler `FC` (Fully Connected) katmanlarına verilir.

### 📉 Baseline Karşılaştırması: Bi-LSTM
- **Kod Nerede?** `src/models/lstm_model.py` içerisinde `LSTMPredictor` sınıfı.
- **Nasıl Çalışır?** Geleneksel olarak `bidirectional=True` ayarıyla çalışır; zaman serisini hem geçmişten geleceğe hem de gelecekten geçmişe okuyarak temporal (zamansal) bağımlılıkları yakalar.
- **Neden Var?** Bi-LSTM, projede Transformer modeli için bir 'baseline' (kıyaslama noktası) görevi görür. Bi-LSTM sadece son zaman adımının gizli durumunu (last hidden state, `lstm_out[:, -1, :]`) kullanırken, Transformer tüm adımları havuzlar (pooling). Mülakatta bu iki mimarinin farkını anlatırken LSTM'in Vanishing Gradient problemine değinebilirsin.

### 📝 FinBERT Kullanımı ve NER (Varlık Çıkarımı)
- **FinBERT (Sentiment):** `src/nlp/sentiment_analyzer.py` içinde. Standart BERT finansal metinleri (örn: "işten çıkarma maliyetleri düşürür") yanlış anladığı için, `ProsusAI/finbert` modeli `AutoModelForSequenceClassification` ile yüklenir. *Batch inference* ve *caching* kullanılarak hız optimize edilir.
- **NER (Entity Extraction):** `src/nlp/entity_extractor.py` içinde. Tüm metni LLM'e verip "Bana ticker'ları bul" demek pahalı olduğu için, `pipeline("ner", model="dslim/bert-base-NER")` ile genel varlıklar bulunur, ekstra olarak Regex (parasalları ve AAPL gibi hisse sembollerini bulmak için) blacklist mantığıyla harmanlanır. Bu iki adım `FinBERT + NER = Context` işlemini en ucuz (computationally cheap) şekilde çözer.

### ⚡ Inference Optimizasyonu: INT8 Dynamic Quantization
- **Kod Nerede?** `src/optimization/quantizer.py` içerisinde `ModelQuantizer` sınıfında.
- **Nasıl Çalışır?** PyTorch'un `torch.quantization.quantize_dynamic` fonksiyonu kullanılarak devasa FP32 (32-bit) ağırlıklar INT8'e (8-bit Integer) dönüştürülür. 
- **Mühendislik Detayı (Önemli):** PyTorch 2.1+ versiyonlarında TransformerEncoder'ın `out_proj.weight` işlemlerinde bir fast-path quantization bug'ı vardır. Kod içerisinde bunu aşmak için özel bir kontrol (`if model_cpu.__class__.__name__ != "TransformerPredictor"`) yazılmıştır. Model Transformer ise `nn.Linear` katmanları dinamik kuantizasyondan hariç tutularak hem ~%60 boyut küçültmesi (RAM tasarrufu) sağlanmış hem de PyTorch hataları bypass edilmiştir.

### 🤖 LLM Fine-Tuning: LoRA mı QLoRA mı?
- **Kod Nerede?** `scripts/finetune_llm.py` içerisinde.
- **Nasıl Yapıldı?** Tamamen **QLoRA** yapıldı. Kodda `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` kullanılarak model (Qwen 2.5) 4-bit donduruluyor. Ardından `LoraConfig(r=16, lora_alpha=32)` ile sadece q_proj, v_proj gibi Attention katmanlarına ağırlık adaptörleri takılıyor.

### 🧠 LangChain Entegrasyonu
- **Kod Nerede?** `src/nlp/llm_rag_chain.py` içerisinde `FinancialRAGSystem` sınıfında.
- **Nasıl Çalışır?** Ollama üzerinden yerel Qwen 2.5 modeli çalıştırılır. LangChain Expression Language (LCEL) kullanılarak (`prompt | self.llm | StrOutputParser()`) bir prompt pipeline'ı kurulur. Sayısal modellerin tahminleri ve FinBERT duyarlılık skorları bu RAG zincirine beslenir.

### 🧬 MLOps ve A/B Testing: MLflow Ne İşe Yarıyor?
- **Kod Nerede?** `src/mlops/ab_testing.py` ve `src/mlops/experiment_tracker.py` içerisinde.
- **Nasıl Çalışır?** `ABTestFramework` sınıfı trafik bazlı yönlendirme yapar. İki modelin (Transformer vs LSTM) hata paylarını (squared errors) toplar ve `scipy.stats.ttest_rel` ile **Paired T-Test** (Eşleştirilmiş T-Testi) uygular. Eğer p-value < 0.05 ise ve hata oranı düşükse modeli promote eder.
- **MLflow Neden Var?** Tüm bu deneyleri, p-value sonuçlarını, hiperparametreleri (learning rate vb.) ve model ağırlıklarını bir UI dashboard üzerinde (Model Registry) versiyonlar ve kayıt altına alır.

---

**Mülakat Tüyosu:** Bu döküman projenin kalbidir. Mülakatçı sana "Transformer'da neden Positional Encoding kullandın?" dediğinde, *"Transformer'ların inherent (doğal) bir sıralama yeteneği olmadığı için, sin/cos fonksiyonlarıyla zaman adımını encode ettim (`PositionalEncoding` sınıfı)*" diyebilirsin. Veya "Quantization sırasında sorun yaşadın mı?" sorusuna *"PyTorch 2.1 fast-path bug'ını bypass ettim"* dersen bu seni %1'lik dilime sokar.
