# اجرای Predict و FastAPI

این بخش برای اجرای CPU نوشته شده است و به Docker وابسته نیست. مدل نهایی پس از آموزش باید در `models/parsbert` قرار بگیرد.

## چیدمان فایل‌های مدل

محتویات پوشهٔ `outputs/parsbert_finetuning/artifacts` در Google Drive را داخل `models/parsbert` کپی کنید. چیدمان نهایی باید این فایل‌ها را داشته باشد:

```text
models/parsbert/
├── model.safetensors
├── model_metadata.json
├── encoder_config/
│   └── config.json
└── tokenizer/
    └── ... فایل‌های ذخیره‌شده توسط tokenizer ...
```

از نوت‌بوک به‌روزشده استفاده کنید تا `encoder_config` همراه مدل ذخیره شود. این کار باعث می‌شود بارگذاری مدل برای inference به دریافت config از Hugging Face وابسته نباشد.

## نصب وابستگی‌ها

در PowerShell و محیط مجازی پروژه:

```powershell
pip install -r requirements-api.txt
```

## Predict از خط فرمان

از ریشهٔ پروژه:

```powershell
python scripts/predict.py "کیفیت محصول خوب بود و از خریدم راضی هستم"
```

اگر مدل جای دیگری است، مسیرش را مشخص کنید:

```powershell
python scripts/predict.py --model-dir "E:\path\to\parsbert" "متن نظر"
```

## اجرای API

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000
```

مستندات تعاملی در `http://127.0.0.1:8000/docs` در دسترس است. مسیرهای اصلی:

- `GET /health`: بررسی روشن بودن سرویس.
- `GET /ready`: بررسی بارگذاری موفق مدل.
- `POST /normalize`: نمایش متن پس از نرمال‌سازی API.
- `POST /predict`: پیش‌بینی هم‌زمان `recommendation_status` و `rating_band`.

نمونهٔ درخواست پیش‌بینی:

```powershell
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/predict" `
  -ContentType "application/json" `
  -Body '{"text":"کیفیت محصول خوب بود و از خریدم راضی هستم"}'
```

اگر فایل‌های مدل هنوز در مسیر مورد انتظار نباشند، سرویس بالا می‌آید و `/health` پاسخ می‌دهد؛ `/ready` و `/predict` تا زمان قرار گرفتن artifactها پاسخ `503` می‌دهند.
