"""Leitor de livros do dataset mystery-crime-books.

Na primeira execução converte o parquet em arquivos JSON (pasta dados/),
depois sobe um servidor local e abre o navegador.

Uso:  python leitor.py            (porta 8765)
      python leitor.py --rebuild  (refaz os dados)
"""
import json
import re
import shutil
import sys
import threading
import unicodedata
import urllib.parse
import urllib.request
import webbrowser
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

AQUI = Path(__file__).resolve().parent
PARQUET = AQUI.parent / "mystery-crime-books" / "data" / "train-00000-of-00001.parquet"
DADOS = AQUI / "dados"
PORTA = 8765

PREFIXO_URL = "https://www.bookrix.com/_ebook-"
TITULO_CAPITULO = re.compile(
    r"^(?i:chapter|cap[ií]tulo|part|book|prologue|epilogue|preface|introduction|"
    r"the adventure of|adventure)\b|^[IVXLC]+(\.|$)|^[IVXLC]+\. [A-Z]|^\d{1,3}(\.|$)"
)

# ---------------------------------------------------------------------------
# Autor e título
#
# A primeira linha de cada texto é "Autor [coautores] Título [subtítulo/dedicatória]".
# O slug da URL é "autor-titulo" (só o primeiro autor), então ele diz exatamente quais
# palavras formam o título; a dúvida é só onde o nome do autor termina.
# ---------------------------------------------------------------------------

# Casos que nenhuma regra resolve (nomes de 3 palavras, títulos ambíguos).
CORRECOES = {
    "amardeep-kaur-randhawa-the-girl-next-door": ("Amardeep Kaur Randhawa", "The Girl Next Door"),
    "katherine-angela-yeboah-lucid-sample-chapters": ("Katherine Angela Yeboah", "Lucid - Sample Chapters"),
    "raymond-beresford-hamilton-solomon-s-ring": ("Raymond Beresford Hamilton", "Solomon's Ring"),
    "melinda-dawn-mardars-satifaction-guaranteed": ("Melinda Dawn Mardars", "Satifaction Guaranteed"),
    "john-kendrick-bangs-r-holmes-co": ("John Kendrick Bangs", "R. Holmes & Co."),
    "m-jade-glock-it-takes-two-to-murder": ("M. Jade Glock", "It Takes Two To Murder"),
    "alyza-in-plain-sight": ("Alyza", "In Plain Sight"),
    "the-general-himself-death-come-swiftly": ("The General Himself", "Death Come Swiftly"),
    "the-general-himself-death-from-the-past": ("The General Himself", "Death from the Past"),
    "the-general-himself-the-face-keeper": ("The General Himself", "The Face Keeper"),
    "dreoilin-seven-sins": ("Dreoilin", "Seven Sins"),
    "tennistar-unforgettable-images": ("Tennistar", "Unforgettable Images"),
    "bethlee-out-of-the-darkness": ("Bethlee", "Out of the Darkness"),
    "me-tommorow-never-comes": ("me", "Tommorow Never Comes"),
    "rina-past-life": ("Rina", "Past Life"),
    "rina-past-life-misian": ("Rina", "Past Life: Misian"),
    "crazykoolaidwriter-search-rescue-solve": ("Crazykoolaidwriter", "Search, Rescue, Solve"),
    "harshavardhan-c-after-death-an-ultimate-truth": ("Harshavardhan C", "After Death, an Ultimate Truth"),
    "brooke-l-private-eye": ("Brooke L", "Private Eye"),
    "sin-c-beautiful-killers": ("Sin .c.", "Beautiful Killers"),
    "james-mad-monk-death-comes": ("James Mad Monk", "Death Comes"),
    "cameron-percy-jeeves-the-old-house": ("Cameron Percy", "Jeeves: The Old House"),
    "cookiebaby-baby-framed-flower": ("Cookiebaby", "Baby Framed Flower"),
    "callele-lyvance-fetiche": ("Callele Lyvance", "Fétiche"),
    "sarah-walker-n-a": ("Sarah Walker", "Sem título"),
    "megan-lewis-sheep-snatcher-thief-catcher-what-8217-s-the-good-in-that-old-thatcher":
        ("Megan Lewis", "Sheep Snatcher, Thief Catcher, What's the Good in That Old Thatcher?"),
    "seren-do-you-believe-in-ghosts": ("Seren", "Do You Believe in Ghosts"),
    "akipana-mindy-fate-sleuth": ("Akipana Mindy", "Fate Sleuth"),
    "chris-harris-grimshaw-amp-mortimer-the-case-book": ("Chris Harris", "Grimshaw & Mortimer: The Case Book"),
    "sola-luna-the-case-of-the-quot-dear-diary-killer-quot": ("Sola Luna", 'The Case of the "Dear Diary Killer"'),
    "grace-cavendish-fabiha-zaman-the-lady-grace-mysteries": ("Grace Cavendish, Fabiha Zaman", "The Lady Grace Mysteries"),
    "bookworm7979-cadena-my-killer": ("Bookworm7979 Cadena", "My Killer"),
    "brewcris-tears-of-the-forgotten": ("Brewcris", "Tears of the Forgotten"),
    "redhead16-marisa-to-love-a-pretty-girl": ("Marisa Redhead16", "To Love a Pretty Girl"),
    "brandonsgirl9951-aka-shannon-why-does-that-crow-follow-me": ("Brandonsgirl9951 (Shannon)", "Why Does That Crow Follow Me?"),
    "jacob-hitchens-im-a-wizard": ("Jacob Hitchens", "Im a Wizard"),
    "lillian-m-valdez-dusk": ("Lillian M. Valdez", "Dusk"),
    "great-a-footprint-made-of-code": ("Great ...", "A Footprint Made of Code"),
}

ENTIDADES = {"039", "8216", "8217", "8220", "8221", "quot", "amp"}
HONORIFICOS = {"sir", "dr", "mr", "mrs", "miss", "lady", "lord"}
PARTICULAS = {"de", "da", "del", "di", "du", "van", "von", "le", "la", "ul", "al", "st", "st."}
SUFIXOS = {"pere", "père", "jr", "sr"}
PREPOSICOES = {"of", "in", "on", "at", "from", "to", "for", "and", "with", "by"}
PALAVRAS_MINUSCULAS = {"a", "an", "the", "of", "in", "on", "at", "to", "for", "and", "or", "but", "by", "with", "from"}


def tokens(texto):
    t = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode().lower()
    return [x for x in re.split(r"[^a-z0-9]+", t) if x and x not in ENTIDADES]


def inicial(p):
    return bool(re.fullmatch(r"\.?(?:[A-Z]\.?){1,3}|[A-Za-z]\.", p)) and p.lower() not in {"the", "a", "an"}


def usuario(p):
    """Apelidos tipo RedKite, writer.juliet, Bookworm7979."""
    return bool(re.search(r"\d|[a-z][A-Z]|[A-Za-z]{2,}\.[A-Za-z]|[A-Za-z]\.[A-Za-z]{2,}", p))


def fim_do_nome(H, i):
    """Consome um nome de pessoa a partir de H[i]; devolve o índice depois dele."""
    n = len(H)
    j = i
    while j < n and inicial(H[j]):
        j += 1
    if j == i:  # começa com primeiro nome
        j += 1
        while j < n and inicial(H[j]):
            j += 1
        while j < n and H[j].lower() in PARTICULAS:
            j += 1
    if j < n and H[j] in {"&", "and"}:  # "David & Stacey Doss", "M.M.K and Aaron Kelley"
        return fim_do_nome(H, j + 1)
    j += 1  # sobrenome
    if j < n and H[j].lower().strip(".,") in SUFIXOS:
        j += 1
    if j < n and H[j] in {"&", "and"}:
        j = fim_do_nome(H, j + 1)
    return j


def proposta_estrutural(H):
    i = 0
    if [w.lower() for w in H[:2]] == ["edited", "by"]:
        i = 2
    elif H and H[0].lower().rstrip(":") == "by":
        i = 1
    if i < len(H) and H[i].lower().rstrip(".") in HONORIFICOS:
        i += 1
    if i < len(H) and usuario(H[i]):
        return i + 1
    return fim_do_nome(H, i)


def proposta_artigo(H, t0):
    """Título que começa com The/A/An logo depois do nome do autor."""
    limite = 2 if usuario(H[0]) else t0 + 2
    for i in range(1, min(len(H), limite)):
        w, antes = H[i], H[i - 1]
        artigo = w in {"The", "THE", "An", "A"} or (w in {"the", "an"} and i <= t0) or (w == "a" and i == 1)
        if artigo and antes.lower() not in PREPOSICOES and not antes.endswith(":"):
            return i
    return None


def proposta_virgula(H, t0):
    """'Nikki Miller, Madi Williams Death to All': coautor depois da vírgula."""
    for i in range(min(t0, 3)):
        if H[i].endswith((",", "/")):
            return i + 1
    return None


class Cabecalho:
    def __init__(self, linha, slug):
        self.H = linha.split()
        self.HT = [tokens(w) for w in self.H]
        S = tokens(slug)
        self.Ss = [S] + ([S[:-1]] if len(S) > 2 and S[-1].isdigit() and len(S[-1]) <= 2 else [])

    def _plano(self, a, b):
        return [t for x in self.HT[a:b] for t in x]

    def fim_titulo(self, t):
        """Se o título pode começar em H[t], devolve onde ele termina."""
        if t <= 0 or t >= len(self.H):
            return None
        for S in self.Ss:
            for e in range(len(self.H), t, -1):
                tit = self._plano(t, e)
                if not tit or len(tit) >= len(S) or S[-len(tit):] != tit:
                    continue
                autor = S[:-len(tit)]
                if any(self._plano(0, a) == autor for a in range(1, t + 1)):
                    return e
        return None

    def valido_perto(self, c):
        for t in list(range(c, len(self.H))) + list(range(c - 1, 0, -1)):
            if self.fim_titulo(t):
                return t
        return None

    def comeca_com(self, autor_tokens):
        for a in range(1, len(self.H)):
            if self._plano(0, a) == autor_tokens:
                return a
        return None


def caixa(texto):
    """Corrige títulos/nomes todos em minúsculas ou todos em maiúsculas."""
    letras = [c for c in texto if c.isalpha()]
    if not letras or not (texto == texto.lower() or (texto == texto.upper() and len(letras) > 3)):
        return texto
    palavras = texto.lower().split()
    return " ".join(p if k and p in PALAVRAS_MINUSCULAS else p[:1].upper() + p[1:] for k, p in enumerate(palavras))


def separar(cabecalhos):
    """Primeira passada com sinais fortes, segunda reaproveitando autores conhecidos."""
    escolhas, conhecidos = [], set()
    for c in cabecalhos:
        t0 = proposta_estrutural(c.H)
        forte = proposta_artigo(c.H, t0)
        if forte and c.fim_titulo(forte):
            conhecidos.add(tuple(c._plano(0, forte)))
            escolhas.append(forte)
            continue
        virgula = proposta_virgula(c.H, t0)
        escolhas.append(c.valido_perto(virgula + 1 if virgula else t0))

    for k, c in enumerate(cabecalhos):
        melhor = None
        for autor in conhecidos:
            a = c.comeca_com(list(autor))
            if a and c.fim_titulo(a) and (melhor is None or a > melhor):
                melhor = a
        if melhor:
            escolhas[k] = melhor

    resultado = []
    for c, t in zip(cabecalhos, escolhas):
        if t is None:  # cabeçalho não bate com o slug (ex.: livros em árabe): só a heurística do nome
            t = min(proposta_estrutural(c.H), len(c.H) - 1)
            resultado.append((" ".join(c.H[:t]), " ".join(c.H[t:]), len(c.H)))
            continue
        e = c.fim_titulo(t)
        autor = re.sub(r"^(by:?|edited by)\s+", "", " ".join(c.H[:t]), flags=re.I) if not c.H[0].lower() == "edited" else " ".join(c.H[:t])
        autor = autor.strip(" ,.~/") or autor
        titulo = re.sub(r"^[^\w'\"(]+", "", " ".join(c.H[t:e]))
        resultado.append((caixa(autor), caixa(titulo), e))
    return resultado


# ---------------------------------------------------------------------------

def limpar(linha):
    return re.sub(r"\s+", " ", linha.replace("�", "").replace("\xa0", " ")).strip()


def paragrafar(linhas):
    paragrafos, capitulos = [], []
    for linha in linhas:
        p = limpar(linha)
        if not p:
            continue
        curto = len(p) <= 70 and not p.endswith((",", ";", "?", "!", '"'))
        if curto and TITULO_CAPITULO.match(p):
            capitulos.append({"t": p, "p": len(paragrafos)})
            paragrafos.append("#" + p)
        else:
            paragrafos.append(p)

    # sumário no início do livro: títulos colados uns nos outros que se repetem depois
    chave = [c["t"].lower().rstrip(".") for c in capitulos]
    def no_sumario(i):
        vizinho = any(0 <= j < len(capitulos) and abs(capitulos[j]["p"] - capitulos[i]["p"]) <= 1
                      for j in (i - 1, i + 1))
        return vizinho and chave[i] in chave[i + 1:]
    capitulos = [c for i, c in enumerate(capitulos) if not no_sumario(i)]
    return paragrafos, capitulos


def trecho_inicial(paragrafos, titulo):
    """Primeiro parágrafo de verdade, pulando dedicatória, capa e sumário."""
    texto = [p for p in paragrafos if not p.startswith("#")]
    bons = [p for p in texto if len(p) >= 120 and not re.match(r"(?i)(dedicat|to my |for my |publication date|copyright|table of contents)", p)]
    escolhido = (bons or texto or [""])[0]
    if escolhido.lower().startswith(titulo.lower()):
        escolhido = escolhido[len(titulo):].lstrip(" .:-")
    return escolhido[:260]


def duplicados(textos):
    """Índices de livros que repetem outro (mesmo conteúdo com cabeçalho diferente)."""
    def amostra(t):
        w = re.findall(r"[a-z]+", t.lower())
        return len(w), {" ".join(w[i:i + 8]) for i in range(0, max(1, len(w) - 8), 50)}
    info = [amostra(t) for t in textos]
    repetidos = set()
    for i in range(len(textos)):
        if i in repetidos or info[i][0] < 20:
            continue
        for j in range(i + 1, len(textos)):
            if j in repetidos or abs(info[i][0] - info[j][0]) > 0.02 * info[i][0] + 5:
                continue
            a, b = info[i][1], info[j][1]
            if len(a & b) >= 0.6 * max(len(a), len(b)):
                repetidos.add(j)
    # mesmo trecho em amostras deslocadas: compara também pelo começo do texto
    return repetidos


def preparar():
    try:
        import pandas as pd
    except ImportError:
        sys.exit("Falta o pandas/pyarrow: rode  pip install pandas pyarrow")
    print("Preparando os livros (só na primeira vez)...")
    df = pd.read_parquet(PARQUET)
    if DADOS.exists():
        shutil.rmtree(DADOS)
    (DADOS / "livros").mkdir(parents=True)

    urls, textos = list(df["url"]), list(df["text"])
    slugs = [u.replace(PREFIXO_URL, "").strip("/") for u in urls]
    linhas = [t.split("\n") for t in textos]
    cabecalhos = [Cabecalho(limpar(l[0]), s) for l, s in zip(linhas, slugs)]
    nomes = separar(cabecalhos)
    repetidos = duplicados(["\n".join(l[1:]) for l in linhas])

    correcoes = {"-".join(tokens(k)): v for k, v in CORRECOES.items()}
    catalogo = []
    for n, (url, slug, ls, cab, (autor, titulo, fim)) in enumerate(zip(urls, slugs, linhas, cabecalhos, nomes)):
        if n in repetidos:
            continue
        chave = "-".join(tokens(slug))
        if chave in correcoes:
            autor, titulo = correcoes[chave]
        resto = " ".join(cab.H[fim:])
        paragrafos, capitulos = paragrafar(([resto] if resto else []) + ls[1:])
        livro_id = f"{n:03d}"
        (DADOS / "livros" / f"{livro_id}.json").write_text(
            json.dumps({"p": paragrafos, "c": capitulos}, ensure_ascii=False), encoding="utf-8"
        )
        catalogo.append({
            "id": livro_id,
            "titulo": titulo or slug.replace("-", " ").title(),
            "autor": autor or "Autor desconhecido",
            "palavras": sum(len(p.split()) for p in paragrafos),
            "capitulos": len(capitulos),
            "url": url,
            "inicio": trecho_inicial(paragrafos, titulo),
        })
    (DADOS / "catalogo.json").write_text(json.dumps(catalogo, ensure_ascii=False), encoding="utf-8")
    print(f"{len(catalogo)} livros prontos ({len(repetidos)} duplicados removidos).")


# ---------------------------------------------------------------------------
# Tradução inglês → português (Google Tradutor, sem chave)
#
# O navegador pede os parágrafos que estão perto da tela; cada tradução fica salva
# em traducoes/<id>.json (fora de dados/, para sobreviver a um --rebuild).
# ---------------------------------------------------------------------------

TRADUCOES = AQUI / "traducoes"
CABECALHO_HTTP = {"User-Agent": "Mozilla/5.0"}
MAX_URL = 6000        # tamanho do pedido em lote (parâmetros já codificados)
MAX_TEXTO = 4000      # parágrafo maior que isso é traduzido em pedaços

_travas = defaultdict(threading.Lock)
_caches, _livros = {}, {}


def _google_lote(textos):
    """Vários parágrafos num pedido; a resposta vem como lista na mesma ordem."""
    q = "&".join("q=" + urllib.parse.quote(t) for t in textos)
    url = "https://clients5.google.com/translate_a/t?client=dict-chrome-ex&sl=en&tl=pt&" + q
    with urllib.request.urlopen(urllib.request.Request(url, headers=CABECALHO_HTTP), timeout=30) as r:
        res = json.loads(r.read())
    saida = [x[0] if isinstance(x, list) else x for x in res]
    if len(saida) != len(textos):
        raise ValueError("resposta desalinhada")
    return saida


def _google_um(texto):
    dados = urllib.parse.urlencode({"q": texto}).encode()
    url = "https://translate.googleapis.com/translate_a/single?client=gtx&sl=en&tl=pt&dt=t"
    with urllib.request.urlopen(urllib.request.Request(url, data=dados, headers=CABECALHO_HTTP), timeout=30) as r:
        return "".join(s[0] for s in json.loads(r.read())[0] if s[0])


def _traduzir_longo(texto):
    frases = re.split(r"(?<=[.!?…])\s+", texto)
    pedacos, atual = [], ""
    for f in frases:
        if atual and len(atual) + len(f) > MAX_TEXTO:
            pedacos.append(atual)
            atual = ""
        atual = f"{atual} {f}".strip()
    pedacos.append(atual)
    return " ".join(_google_um(p[:MAX_TEXTO * 2]) for p in pedacos)


def traduzir_textos(textos):
    resultado = [None] * len(textos)
    lotes, lote, tamanho = [], [], 0
    for i, t in enumerate(textos):
        if len(t) > MAX_TEXTO:
            resultado[i] = _traduzir_longo(t)
            continue
        custo = len(urllib.parse.quote(t)) + 3
        if lote and (tamanho + custo > MAX_URL or len(lote) >= 50):
            lotes.append(lote)
            lote, tamanho = [], 0
        lote.append(i)
        tamanho += custo
    if lote:
        lotes.append(lote)

    def fazer(lote):
        try:
            for i, t in zip(lote, _google_lote([textos[i] for i in lote])):
                resultado[i] = t
        except ValueError:  # desalinhou: um por um
            for i in lote:
                resultado[i] = _google_um(textos[i])

    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(fazer, lotes))
    return resultado


def _paragrafos(livro_id):
    if livro_id not in _livros:
        _livros[livro_id] = json.loads((DADOS / "livros" / f"{livro_id}.json").read_text(encoding="utf-8"))["p"]
    return _livros[livro_id]


def _cache(livro_id):
    if livro_id not in _caches:
        arq = TRADUCOES / f"{livro_id}.json"
        _caches[livro_id] = json.loads(arq.read_text(encoding="utf-8")) if arq.exists() else {}
    return _caches[livro_id]


def traduzir_paragrafos(livro_id, indices):
    ps = _paragrafos(livro_id)
    with _travas[livro_id]:
        cache = _cache(livro_id)
        faltam = sorted({i for i in indices if 0 <= i < len(ps) and str(i) not in cache})
    if faltam:
        traduzidos = traduzir_textos([ps[i].lstrip("#") for i in faltam])
        with _travas[livro_id]:
            cache.update({str(i): t for i, t in zip(faltam, traduzidos)})
            TRADUCOES.mkdir(exist_ok=True)
            tmp = TRADUCOES / f"{livro_id}.json.tmp"
            tmp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
            tmp.replace(TRADUCOES / f"{livro_id}.json")
    return {i: cache[str(i)] for i in indices if str(i) in cache}


class Handler(SimpleHTTPRequestHandler):
    def end_headers(self):
        # sem cache: depois de um --rebuild o navegador precisa pegar os dados novos
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def _json(self, status, obj):
        corpo = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(corpo)))
        self.end_headers()
        self.wfile.write(corpo)

    def do_POST(self):
        if self.path != "/api/traduzir":
            return self.send_error(404)
        try:
            pedido = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            livro_id = str(pedido["id"])
            if not re.fullmatch(r"\d{3}", livro_id) or not (DADOS / "livros" / f"{livro_id}.json").exists():
                return self._json(404, {"erro": "Livro não encontrado"})
            indices = [int(i) for i in pedido["indices"]][:120]
        except (ValueError, KeyError, TypeError):
            return self._json(400, {"erro": "Pedido inválido"})
        try:
            self._json(200, {"traducoes": traduzir_paragrafos(livro_id, indices)})
        except (OSError, ValueError) as e:  # sem internet, bloqueio do serviço etc.
            self._json(502, {"erro": f"Serviço de tradução indisponível ({e.__class__.__name__})"})

    def log_message(self, *args):
        pass


def main():
    if "--rebuild" in sys.argv or not (DADOS / "catalogo.json").exists():
        preparar()
    servidor = ThreadingHTTPServer(("127.0.0.1", PORTA), partial(Handler, directory=str(AQUI)))
    endereco = f"http://127.0.0.1:{PORTA}/"
    print(f"Leitor aberto em {endereco}  (Ctrl+C para fechar)")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, webbrowser.open, [endereco]).start()
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
