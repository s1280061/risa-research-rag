FROM python:3.11-slim
# → ベースとなるOSを指定（Python3.11入りの軽量Linux）

WORKDIR /app
# → コンテナ内の作業フォルダを /app に設定

COPY requirements.txt .
# → ローカルのrequirements.txtをコンテナにコピー

RUN pip install --no-cache-dir -r requirements.txt
# → コンテナ内でpip installを実行

COPY rag_pipeline.py .
# → ローカルのPythonファイルをコンテナにコピー

CMD ["python", "rag_pipeline.py"]
# → コンテナ起動時に実行するコマンド