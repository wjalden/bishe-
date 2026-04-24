from fastapi import FastAPI

app = FastAPI(title='HTC Demo API')


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/predict')
def predict(payload: dict):
    text = payload.get('text', '')
    return {
        'text': text,
        'predictions': [
            {'label': 'L0', 'score': 0.72, 'path': ['ROOT', 'L0']},
            {'label': 'L4', 'score': 0.41, 'path': ['ROOT', 'L0', 'L4']}
        ]
    }
