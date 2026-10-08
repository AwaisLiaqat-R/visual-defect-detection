# DefectVision AI

A small service that looks at a photo of a manufactured part and tells you whether it's fine or defective. You upload an image, and it answers in well under a second.
## What this project does

Picture a factory producing thousands of metal parts an hour. Every so often a scratched or cracked one slips down the line. A person inspecting by eye gets tired and can realistically check about one part per second.

This model handles roughly 78 images per second when run in batches, and a single image comes back in about 47 ms. You send a photo, and the reply is either "normal" or "defective", along with how confident the model is.

## Results

| Metric | Score | What it means |
|:---|:---:|:---|
| Overall accuracy | 96.4% | About 96 correct answers out of every 100 parts |
| Defect detection rate | 92.5% | Catches roughly 92 of every 100 defective parts |
| False alarms | Low | Good parts are rarely flagged by mistake |
| Latency | ~47 ms | Per image, on a regular CPU |

Raw accuracy isn't the number we care about most. Shipping a broken part to a customer costs far more than pulling a good part aside for a second look, so we treated a missed defect as about ten times worse than a false alarm. The model is deliberately tuned to be cautious: it would rather flag a few good parts than let a bad one through.

## How it works

We didn't train a network from scratch. We started with ResNet-18, which was already pre-trained on a large general image set and so already understands edges, textures and shapes. Then we fine-tuned it on photos of surface defects.

This approach has a few practical benefits:

- Training takes hours instead of days.
- It works well even with a fairly small dataset.
- It runs on an ordinary CPU, so no GPU is needed for inference.

The intended setup on a production line looks like this:
Camera takes a photo of the part
       
ResNet-18 scans the surface for scratches, cracks and fractures

Model returns "normal" or "defective"

Sorting machine acts on the result

Good part continues  |  Defective part is set aside for review

## Two kinds of mistakes

Mistake What happens Rough cost Missed defect (false negative) A broken part reaches the customer $10

False alarm (false positive) A good part goes to manual review $1

Because the first mistake is so much more expensive, we lowered the decision threshold so the model flags a part as defective more readily. Tuning the threshold this way improved results on both counts:

Default threshold Tuned threshold 
Accuracy | 96.8% | 97.4% |

Defect catch rate | 95.8% | 98.4% |

Cost per 1,000 parts | Higher | Lower |

### Project Layout

```text
.
├── src/
│   ├── models.py
│   ├── train.py
│   ├── evaluate.py
│   └── infer.py
├── api/
│   └── app.py
├── checkpoints/
│   └── best_model.pth
├── vercel-demo/
│   └── index.html
├── tests/
├── Dockerfile
├── render.yaml
└── README.md
```

#### File Descriptions
* **`src/models.py`**: ResNet-18 architecture.
* **`src/train.py`**: Model training loop.
* **`src/evaluate.py`**: Evaluation script and accuracy metrics.
* **`src/infer.py`**: Inference code used by the API backend.
* **`api/app.py`**: FastAPI application exposing `/predict`, `/batch-predict`, `/health`, and `/metrics`.
* **`checkpoints/best_model.pth`**: Trained model weights (~128 MB).
* **`vercel-demo/index.html`**: Frontend drag-and-drop web demo page.
* **`tests/`**: Automated unit and integration tests.
* **`render.yaml`**: Infrastructure-as-code deployment configuration for Render.com.
* **`Dockerfile`**: Containerization setup for production.


### 2. Install dependencies
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac/Linux

pip install -r requirements.txt


### 3. Start the API
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
Open [http://localhost:8000/docs](http://localhost:8000/docs) and you'll get an interactive page where you can upload images and try the endpoints from your browser.

### 4. Or use Docker

docker-compose up --build

This gives you the same result without setting up Python yourself.


## API reference

**`GET /health`**
Confirms the service is running. Returns `{ "status": "healthy", "model_loaded": true }`.

**`POST /predict`**
Analyses a single image. Send form-data with the image in a field called `file`.

**`POST /batch-predict`**
Analyses several images in one request. Send form-data with multiple `files`. Useful for checking a whole batch of parts at once.

**`GET /metrics`**
Usage statistics: total inspections, defect rate and average response time.

## Deployment

The project is deployed in two parts:

| Part | Platform | URL |
|:---|:---|:---|
| API and model (FastAPI) | Render.com | https://defect-detection-api.onrender.com |
| Demo website | Vercel | https://defect-vision-demo.vercel.app |

If you want to host your own copy, `render.yaml` covers the backend and `vercel-demo/vercel.json` covers the demo site.


## Known limitations

**Lighting.** The model was trained under one set of lighting conditions. If your factory is lit very differently, accuracy can drop. The fix is to retrain or fine-tune with photos from your own line.

**It says whether, not where.** The output is a yes/no answer. It doesn't mark the location of the defect. Drawing a box around the flaw would be a sensible next step.

**Slow first request on the free tier.** Render's free plan puts the service to sleep after 15 minutes without traffic. The first request after that can take about 30 seconds while it wakes up. Everything after that is fast.

## Demo walkthrough

If you're presenting the project, this is the order we used:

| Time | On screen | What's happening |
| 0:00 - 0:35 | Dataset samples | About 3,000 labelled surface photos used for training |
| 0:35 - 1:20 | Two models training side by side | A custom CNN against pre-trained ResNet-18; ResNet-18 comes out ahead |
| 1:20 - 2:00 | Confusion matrix and threshold plot | Tuning the threshold to catch more defects |
| 2:00 - 2:45 | Live demo in the browser | Upload a photo and get a result in under 50 ms |


## Built with

| Tool | Used for |
| PyTorch and ResNet-18 | The image classification model |
| FastAPI | The web server that accepts uploads |
| Docker | Packaging, so it runs the same everywhere |
| Render.com | Hosting the backend |
| Vercel | Hosting the demo page |
| KolektorSDD2 | Roughly 3,000 labelled industrial surface images used for training |
