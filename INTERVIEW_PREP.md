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

### 📝 FinBERT Kullanımı: Nerede ve Neden?
- **Kod Nerede?** `src/nlp/sentiment_analyzer.py` içerisinde.
- **Nasıl Çalışır?** `ProsusAI/finbert` modeli `AutoModelForSequenceClassification` ile yüklenir. GPU/MPS tespiti otomatik yapılır. `analyze_batch` fonksiyonunda haberler önbellek (cache) mimarisiyle batch (toplu) olarak işlenir.
- **Neden Seçildi?** Standart BERT, finansal metinleri yanlış anlar. Örneğin "Hisse %5 düştü" cümlesi standart BERT için nötr olabilirken, FinBERT bunu 50.000 finans makalesiyle eğitildiği için doğru şekilde "Negatif" olarak sınıflandırır. Bu, sayısal verimi haberlerle hizalamak (alignment) için şarttır.

### 🔍 Varlık Tanıma (NER): Nerede ve Nasıl?
- **Kod Nerede?** `src/nlp/entity_extractor.py` içerisinde.
- **Nasıl Çalışır?** İkili bir sistem vardır. 1. `pipeline("ner", model="dslim/bert-base-NER")` ile genel ORG, PER, LOC varlıkları bulunur. 2. `re.compile` kullanılarak finansal (ticker, $, milyar/milyon) ifadeleri regex fallback ile çıkarılır (örn. blacklist filtresiyle İngilizce bağlaçlar elenir).
- **Neden Bu Yaklaşım?** Saf bir LLM'e tüm metni verip "Bana ticker'ları bul" demek çok pahalı (token cost) ve yavaştır. Bert-base-NER ve Regex kombinasyonu deterministik, ucuz ve sub-millisecond hızında çalışır.

### 🤖 LLM Fine-Tuning: LoRA mı QLoRA mı?
- **Kod Nerede?** `scripts/finetune_llm.py` içerisinde.
- **Nasıl Yapıldı?** Tamamen **QLoRA** yapıldı. Kodda `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")` kullanılarak model (Qwen 2.5) 4-bit donduruluyor. Ardından `LoraConfig(r=16, lora_alpha=32)` ile sadece q_proj, v_proj gibi Attention katmanlarına ağırlık adaptörleri takılıyor.
- **Neden QLoRA?** 7 Milyar parametreli bir modeli (7B) tam (full) fine-tune etmek için devasa A100 GPU'lar gerekir. QLoRA sayesinde sadece adaptörleri (milyonda bir parametre) tüketici sınıfı (RTX 3090 vb.) bir GPU'da 24GB VRAM sınırını aşmadan eğitebiliyorum.

### 🧠 LangChain Entegrasyonu: Nerede ve Neden?
- **Kod Nerede?** `src/nlp/llm_rag_chain.py` içerisinde `FinancialRAGSystem` sınıfında.
- **Nasıl Çalışır?** `ChatPromptTemplate.from_messages` kullanılarak `system` (analist personası) ve `human` (gelen dinamik metrikler) promptları ayrılır. Ollama üzerinden yerel olarak çalışan modele LangChain Expression Language (LCEL) kullanılarak (`prompt | self.llm | StrOutputParser()`) bağlanır.
- **Neden LangChain?** LLM API'leri değişkendir. LangChain sayesinde yarın Qwen yerine OpenAI veya Anthropic'e geçmek istersem sadece `ChatOllama` sınıfını `ChatOpenAI` ile değiştirmem yeterlidir. Prompt injection riskini ve RAG context'ini yönetmek için en stabil framework'tür.

### 🧬 MLOps ve A/B Testing: MLflow Ne İşe Yarıyor?
- **Kod Nerede?** `src/mlops/ab_testing.py` ve `src/mlops/experiment_tracker.py` içerisinde.
- **Nasıl Çalışır?** `ABTestFramework` sınıfı; trafik bazlı (örn. %90 Champion model, %10 Challenger model) yönlendirme yapar. İki modelin aynı verideki tahmin hatalarını (squared errors) toplar ve `scipy.stats.ttest_rel` ile **Paired T-Test** (Eşleştirilmiş T-Testi) uygular. Eğer p-value < 0.05 ise ve hata oranı daha düşükse yeni modeli "PROMOTE" eder.
- **MLflow Neden Var?** Model denemelerini (örneğin LSTM vs Transformer) sadece konsolda görmek yetmez. MLflow (`training_args(report_to="mlflow")`); denediğim hiperparametreleri (learning rate, epoch vb.), modelin ağırlıklarını ve test metriklerini bir UI dashboard üzerinde kayıt altına alır (Model Registry). Bu, projenin sadece bir "Jupyter Notebook" denemesi olmadığını, tamamen **Bilinçli bir MLOps Mimarisi** olduğunu kanıtlar.

---

**Mülakat Tüyosu:** Bu döküman projenin kalbidir. Mülakatçı sana "FinBERT'i nasıl kullandın?" dediğinde, *"HuggingFace pipeline'ı ile aldım"* demek yerine *"src/nlp/sentiment_analyzer.py içinde AutoModelForSequenceClassification ile yükledim, performansı artırmak için 200 karakterlik key'ler ile önbelleğe (cache) aldım ve analizleri batch (toplu) çalıştırarak GPU verimliliğini artırdım"* dersen, karşı taraf senin gerçekten üretim (production) odaklı, kod kalitesi yüksek bir AI mühendisi olduğunu anlar.
