import pygraphviz as pgv
from 求导.Exp import E


def _drawNode(E, G):
    if E == None:
        return None
    # G.add_node(str(E.index)+' '+E.f)
    my = '[' + str(E.index) + '] ' + str(E.f)
    G.add_node(E.index, label=str(E.f))
    ln = _drawNode(E.left, G)
    rn = _drawNode(E.right, G)
    if ln != None:
        G.add_edge(E.index, ln)
    if rn != None:
        G.add_edge(E.index, rn)

    return E.index


def drawE(E):
    G = pgv.AGraph(directed=True, strict=True)

    _drawNode(E, G)
    G.graph_attr['epsilon'] = '0.01'
    G.write('fooOld.dot')
    G.layout('dot')  # layout with dot
    G.draw('a.png')  # write to file
