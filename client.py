import http.client


for i in range(100):
    conn = http.client.HTTPConnection('118.70.126.72', port=8080)
    conn.request("GET", str(i))
    conn.close