# Ubuntuの最新版をベースイメージとして使用
FROM ubuntu:latest

# 必要なパッケージのインストール
# curlとgitはRustのインストールとzpreztoのクローンに必要
# build-essentialはC言語のコンパイルに必要なツールを含む
# zshはzpreztoを使用するために必要
RUN apt-get update && apt-get install -y curl git build-essential zsh libssl-dev pkg-config graphviz

# Rustのインストール
# ここではRustの公式インストールスクリプトを使用
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y

# 環境変数PATHにRustのパスを追加
# Dockerイメージ内で新しいシェルセッションを開始したときにRustが利用可能になる
ENV PATH="/root/.cargo/bin:${PATH}"

# Preztoのインストール
RUN zsh -c "\
    git clone --recursive https://github.com/sorin-ionescu/prezto.git '${ZDOTDIR:-$HOME}/.zprezto'; \
    setopt EXTENDED_GLOB; \
    for rcfile in '${ZDOTDIR:-$HOME}'/.zprezto/runcoms/^README.md(.N); do \
    ln -s \"\$rcfile\" \"${ZDOTDIR:-$HOME}/.\${rcfile:t}\"; \
    done; \
"
# .zpreztorcをコピー（もしカスタマイズした.zpreztorcがあれば）
COPY .zpreztorc /root/.zpreztorc

# 作業ディレクトリの設定（任意）
WORKDIR /workspace

# コンテナ起動時に実行されるコマンド
# ここではzshを起動する
CMD ["zsh"]
