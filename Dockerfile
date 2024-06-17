# Ubuntuの最新版をベースイメージとして使用
FROM ubuntu:latest

RUN apt-get update && apt-get install -y curl git build-essential zsh libssl-dev pkg-config graphviz openssh-client

# Rustのインストール
# ここではRustの公式インストールスクリプトを使用
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 環境変数PATHにRustのパスを追加
# Dockerイメージ内で新しいシェルセッションを開始したときにRustが利用可能になる
ENV PATH="/root/.cargo/bin:${PATH}"

# 作業ディレクトリの設定（任意）
WORKDIR /workspace

# コンテナ起動時に実行されるコマンド
CMD ["bash"]
