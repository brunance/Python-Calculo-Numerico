# 🧮 EP1 — Cálculo Numérico

## 🎯 Objetivo
Gerar os **mapas de convergência (fractais)** dos três sistemas não lineares fornecidos no enunciado, utilizando o **método de Newton**.

---

## 📁 Estrutura do projeto

```
EP1/
│
├── ep1_luiz_henrique.py    # Código principal com os três sistemas
├── sistema_1.png           # Mapa de convergência do Sistema 1
├── sistema_2.png           # Mapa de convergência do Sistema 2
├── sistema_3.png           # Mapa de convergência do Sistema 3
└── README.md               # Instruções (este arquivo)
```

---

## ⚙️ Pré-requisitos

- Python 3 instalado (versão 3.8 ou superior)
- Bibliotecas:
  - `numpy`
  - `matplotlib`

---

## 🧩 Instalação das dependências

No terminal, dentro da pasta do projeto, execute:

```bash
pip install numpy matplotlib
```

---

## ▶️ Como executar o programa

1. Certifique-se de que o arquivo `ep1_luiz_henrique.py` está na pasta do projeto.  
2. No terminal, execute:

```bash
python ep1_luiz_henrique.py
```

3. O programa irá:
   - Gerar os gráficos dos **três sistemas**.
   - Exibir as imagens na tela.
   - Salvar automaticamente cada figura como:
     - `sistema_1.png`
     - `sistema_2.png`
     - `sistema_3.png`

---

## 📊 Resultados esperados

- Cada imagem representa o **mapa de convergência** de um sistema.  
- As **cores** indicam para **qual raiz** cada ponto inicial convergiu.  
- As **fronteiras complexas** entre as regiões coloridas formam o **fractal de Newton**.

---

## 🧾 Relatório (entrega)

O relatório deve conter:
1. Nome do(s) aluno(s)  
2. Objetivo do trabalho (pode copiar o enunciado)  
3. As três figuras geradas (`sistema_1.png`, `sistema_2.png`, `sistema_3.png`)  
4. Uma breve descrição, por exemplo:

> As regiões coloridas representam os pontos iniciais que convergem para diferentes raízes do sistema.  
> As fronteiras complexas observadas correspondem à estrutura fractal do método de Newton.

---

## 👨‍💻 Autor

**Luiz Henrique Monteiro**  
Cálculo Numérico — EP1 — 2025
