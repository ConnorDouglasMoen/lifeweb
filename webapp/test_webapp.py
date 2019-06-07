# -*- coding: utf-8 -*-
from flask import Flask, request, render_template

app = Flask(__name__)

@app.route("/")
def test():
	return render_template('test.html')
