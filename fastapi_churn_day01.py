from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def root():
    return {"message": "ml churn service is running"}
