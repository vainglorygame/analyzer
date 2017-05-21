#!/usr/bin/python3
import csv
import functools

from flask import (
    Flask,
    render_template,
    send_from_directory,
    jsonify
)

app = Flask(__name__)

# decision tree
class Node(object):
    def __init__(self, name="", count=0):
        self.children = []
        self.name = name
        self.count = count

    def child(self, name, count):
        c = Node(name, count)
        self.children.append(c)
        return c

    def get(self, name):
        return [c for c in self.children if c.name == name][0]

    def traverse(self, path):
        cur = self
        for child in path:
            cur = cur.get(child)
        return cur

    def predict(self, *path):
        par = self.traverse(list(path))
        return [{
            "name": p.name,
            "probability": p.count/par.count
        } for p in par.children]

tree = Node()

def load():
    def deeper(depth=1, parent=tree, max=8):
        for depth in range(max):
            with open("data/" + str(depth + 2) + ".csv") as f:
                fr = csv.reader(f, delimiter=",", quotechar="\"")
                for row in fr:
                    path = row[:-2]
                    name = row[-2]
                    c = int(row[-1])
                    parn = parent.traverse(path)
                    parn.child(name, c)

    deeper()
    tree.count += sum(c.count for c in tree.children)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/assets/<path:path>")
def assets(path):
    return send_from_directory("assets", path)

@app.route("/api/builds")
@app.route("/api/builds/<path:path>")
def api(path=None):
    if path == None:
        els = []
    else:
        els = path.split("/")
    return jsonify(tree.predict(*els))

if __name__ == "__main__":
    load()
    app.run(debug=True)
