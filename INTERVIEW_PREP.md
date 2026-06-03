# StockSense AI: LLM/NLP/AI Engineer Mülakat Hazırlık Rehberi 🚀

Bu doküman, **StockSense AI** projesini bir **AI/NLP/LLM Engineer** mülakatında en güçlü şekilde savunabilmen, mimari kararlarının altını doldurabilmen ve gelebilecek zorlu teknik sorulara karşı hazırlıklı olman için tasarlanmıştır.

---

## 1. Proje Özeti ve Teknik Mimari (Senin Dilinden)

**"Mülakatta projeni nasıl anlatmalısın?"**

> "StockSense AI, sadece tarihsel fiyat verilerine odaklanan geleneksel finansal modellerin eksikliğini gidermek amacıyla geliştirdiğim production-ready bir 'Zeka Döngüsü' (Intelligence Lifecycle) projesidir.
> 
> Çoğu sistem ya sadece fiyat (OHLCV) tahmini yapar ya da sadece haber duyarlılığı (sentiment) çıkarır. Ben bu iki dünyayı sentezledim. Sistemde volatil finansal zaman serilerini modellemek için **Multi-Head Self-Attention** kullanan tamamen **Custom bir PyTorch Transformer Encoder** mimarisi tasarladım. Eş zamanlı olarak, finansal metinleri analiz etmek için **FinBERT** tabanlı bir NLP ardışık düzeni (pipeline) kurdum ve Kural Tabanlı NER (Named Entity Recognition) entegre ettim.
> 
> En büyük katma değerim ise bu karmaşık nicel (quantitative) ve nitel (qualitative) çıktıları **LangChain** ve yerel çalışan **Qwen 2.5 LLM** kullanarak bir RAG ajanında birleştirmek oldu. Sistem, sadece '- %2 düşecek' demek yerine bir yatırım uzmanı gibi gerekçeli akademik raporlar üretiyor. Son olarak, üretim (production) aşamasında modeli **INT8 Dynamic Quantization** ile sıkıştırıp **ONNX Runtime**'a taşıyarak sub-millisecond (milisaniyenin altında) bir gecikme (latency) seviyesine indirdim."

---

## 2. Kullanılan Teknolojiler ve "Neden?" Soruları

### 🧠 Derin Öğrenme: Neden LSTM yerine Custom Transformer Encoder?
- **Tercih Nedeni:** Zaman serisi analizinde LSTM'ler standarttır ancak uzun vadeli bağımlılıkları (long-term dependencies) yakalamakta zorlanırlar (Vanishing Gradient problemi) ve sıralı (sequential) çalıştıkları için yavaştırlar. 
- **Savunma:** "Volatil borsa verilerindeki ani değişimleri yakalamak için Attention mekanizması şarttı. PyTorch'un kendi `nn.TransformerEncoder` katmanını, Pre-Normalization ve Sinusoidal Positional Encoding kullanarak sıfırdan uyarladım. Bu sayede modelin verinin tüm tarihsel bağlamına aynı anda bakmasını (Context-aware Global Dependencies) ve paralel eğitim alabilmesini sağladım."

### 📝 NLP: Neden Standart BERT veya LLM yerine FinBERT?
- **Tercih Nedeni:** Domain adaptation (Alan uyarlaması).
- **Savunma:** "Standart bir BERT modeli 'Şirket işten çıkarma yaptı' cümlesini negatif bir duygu (sentiment) olarak sınıflandırır. Ancak finans dünyasında işten çıkarmalar genellikle maliyet azaltımı olarak görülür ve hisse fiyatını olumlu etkileyebilir. Bu yüzden özel olarak finansal veri setleriyle eğitilmiş ProsusAI imzalı **FinBERT**'i kullandım. Bu, nicel sinyallerimle (quantitative signals) gerçek dünya haberlerini birbirine çok daha doğru hizalamamı (alignment) sağladı."

### 🤖 LLM & RAG: Neden LangChain ve Yerel Qwen 2.5?
- **Tercih Nedeni:** Finansal stratejiler üretirken veri gizliliği (privacy) kritiktir.
- **Savunma:** "Kullanıcının portföy veya hisse analiz verilerini OpenAI gibi dış API'lere göndermek istemedim. **Qwen 2.5 (7B-Instruct)**, kodlama ve mantıksal akıl yürütme (reasoning) konusunda yerel çalışabilen en başarılı açık kaynaklı modellerden biri. LangChain ile RAG motorunu tasarlayarak, sayısal modelimin sonuçlarını ve FinBERT'in duyarlılık skorlarını LLM'e zenginleştirilmiş bağlam (Augmented Context) olarak sundum."

### ⚙️ Optimizasyon: Neden INT8 Quantization ve ONNX?
- **Tercih Nedeni:** PyTorch modelleri üretim (production) ortamında API üzerinden (FastAPI) sunulurken yavaştır ve fazla kaynak tüketir.
- **Savunma:** "Post-Training Dynamic Quantization (INT8) uyguladım. Bunu yaparken PyTorch'un `fast-path` hatalarını aşmak için Transformer dışındaki katmanlara dinamik olarak odaklandım. Doğruluğu (accuracy) kaybetmeden modelin VRAM ve disk ayak izini ~%60 küçülttüm. Ardından model grafiklerini (graph) ONNX formatına çevirip ONNX Runtime üzerinde çalıştırdım. Bu mühendislik eforu, FastAPI endpoint'lerimde inanılmaz bir hız (sub-millisecond latency) sağladı."

---

## 3. Mülakat Simülasyonu: Zorlu Sorular ve İdeal Cevaplar

### Soru 1: LLM Fine-Tuning stratejini anlatır mısın? Projede PEFT (QLoRA) kullanma mantığın neydi?
**Beklenen Derinlik:** LLM eğitim maliyetlerini ve alignmet süreçlerini bilip bilmediğini test etmek.
**Senin Cevabın:** 
> "Büyük dil modellerini (LLM) sıfırdan (Full Fine-tuning) finansal domain'e hizalamak (alignment) ciddi bir donanım gerektirir. Ben repo içerisindeki `finetune_llm.py` dosyasında **PEFT (Parameter-Efficient Fine-Tuning)** yaklaşımını modelledim. Özellikle QLoRA kullanarak modelin baz (base) ağırlıklarını 4-bit dondurup (freeze), sadece ufak adaptör matrisleri (LoRA weights) eğittim. Bu sayede modelin genel yeteneklerini (catastrophic forgetting'i engelleyerek) korurken, çok düşük bir GPU VRAM ile onu finansal raporlama konusunda uzmanlaştırmış oldum."

### Soru 2: Custom Transformer mimarisinde "Sinusoidal Positional Encoding" kullanmışsın. Neden "Learnable Positional Embeddings" (öğrenilebilir konum kodlamaları) kullanmadın?
**Senin Cevabın:** 
> "Zaman serisi (time-series) verilerinde, özellikle finans gibi volatil ve periyodik (örneğin günlük, haftalık trendler) verilerde, Sinusoidal kodlamalar doğal bir frekans ve periyodiklik bilgisi taşır. Learnable (öğrenilebilir) embedding'ler NLP'de başarılı olsa da, eğitim verisinde görünmeyen (unseen) uzunluktaki zaman serilerinde genelleme (extrapolation) yaparken zorlanırlar. Sinusoidal fonksiyonlar ise deterministiktir ve modelin zaman adımları arasındaki göreceli mesafeyi daha iyi anlamasına yardımcı olur."

### Soru 3: MLOps süreçlerini nasıl tasarladın? Modelin "daha iyi" olduğuna nasıl karar veriyorsun?
**Senin Cevabın:** 
> "Yalnızca test setindeki MSE/RMSE gibi metriklerle yetinmiyorum. Pipeline'da **Paired T-Tests (p-value < 0.05)** kullanarak istatistiksel A/B testleri yaptım. Yani yeni eğittiğim Transformer modelinin Bi-LSTM baseline'ından gerçekten istatistiksel olarak (şans eseri değil) anlamlı derecede daha iyi olup olmadığını doğruladım. Tüm bu metrikleri, hiperparametreleri ve model ağırlıklarını da **MLflow** ile Model Registry üzerinde versiyonlayıp takip ettim."

---

## 💡 Gelecek Vizyonu (Mülakatta Ekstra Puan Getirecek Konular)

Mülakatı sen yönlendirmek istersen şu projeksiyonları sunabilirsin:

1. **"Gelecekte DDP (Distributed Data Parallel) eklemeyi planlıyorum:"** 
   - "Şu an model tek GPU/CPU'da eğitilebiliyor. Tick-level (salise bazlı) devasa borsa verileriyle çalışmak için PyTorch DDP entegrasyonu ile modeli çoklu GPU'larda eğitecek bir altyapıya geçmeyi düşünüyorum."
2. **"TensorRT Entegrasyonu:"** 
   - "ONNX geçişi gecikmeyi çok düşürdü ancak NVIDIA GPU kullanan sunucularda maksimum verimi almak için (intra-GPU sub-batch latencies) grafiği TensorRT'ye derlemeyi planlıyorum."
3. **"Agentic Tool Calling (Function Calling):** 
   - "Şu an LLM RAG tabanlı bilgi sentezi yapıyor. Gelecekte LLM'in, örneğin 'X hissesi için RSI hesapla' dediğimde Python REPL veya özel finans API'lerini tetikleyebileceği bir Agentic altyapı (ReAct prompt) kurmayı hedefliyorum."

---

**Mülakat Tüyosu:** Görüşme sırasında; modeli sıfırdan yazabilen bir ML mühendisi (PyTorch Transformer, MLOps, Quantization) ile sistemleri birleştiren bir AI/NLP mühendisi (FinBERT, LangChain, Qwen 2.5) şapkalarını aynı anda takabildiğini (Full-Stack AI Engineering) çok net hissettir. Başarılar! 🏆
