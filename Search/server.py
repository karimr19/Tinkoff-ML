from flask import Flask, render_template, request
from search import score, retrieve, build_index
from time import time


app = Flask(__name__, template_folder='.')
inverted_index, data, word_2_vec, stop_words = build_index()


@app.route('/', methods=['GET'])
def index():
    start_time = time()
    query = request.args.get('query')
    if query is None:
        query = ''
    documents = retrieve(query, inverted_index, data, stop_words)
    scored = [(doc, score(query, doc, word_2_vec)) for doc in documents]
    scored = sorted(scored, key=lambda doc: -doc[1])
    results = [[str(doc), str(doc)] + ['%.2f' % scr] for doc, scr in scored]
    return render_template(
        'index.html',
        time="%.2f" % (time()-start_time),
        query=query,
        search_engine_name='Yandex',
        results=results
    )


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=80)
