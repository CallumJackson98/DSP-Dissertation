from package1 import app
import webbrowser
import random, threading, webbrowser

if __name__ == '__main__':
    
    port = 5000 + random.randint(0, 999)
    url = "http://127.0.0.1:5000/".format(port)
    
    threading.Timer(1.25, lambda: webbrowser.open(url) ).start()
    
    app.run(debug=False)