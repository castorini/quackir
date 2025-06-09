from pyserini.analysis import Analyzer, get_lucene_analyzer

analyzer = Analyzer(get_lucene_analyzer())

def tokenize(to_tokenize):
    return ' '.join(analyzer.analyze(to_tokenize))