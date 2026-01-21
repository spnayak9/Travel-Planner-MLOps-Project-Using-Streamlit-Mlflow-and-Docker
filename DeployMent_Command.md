```bash
py --list
py -3.10 -m venv .venv
.venv\Scripts\activate
python --version
python -m pip install --upgrade pip
pip install -r requirements.txt
Streamlit run app.py
```

```docker
docker --version
docker build -t travel-planner-streamlit .
docker run -p 8501:8501 travel-planner-streamlit
http://localhost:8501
docker ps
docker stop container id
```