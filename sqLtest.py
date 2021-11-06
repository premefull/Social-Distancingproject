import pymysql as m
import datetime




c = m.connect(host='localhost', user='root', passwd='', db='test')
cur = c.cursor()
count = cur.execute("INSERT INTO risk_transaction (numall_stu, num_of_picture) VALUES (8, 20)")
data = cur.fetchone()

c.commit()
cur.close()
c.close()
