import requests
import random
import datetime
import os

headers_list = [
    {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
        'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G955U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/87.0.4280.141 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; SM-G981B) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/80.0.3987.162 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 13_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) CriOS/87.0.4280.77 Mobile/15E148 Safari/604.1',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.109 Safari/537.36 CrKey/1.54.248666',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (X11; Linux aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.188 Safari/537.36 CrKey/1.54.250320',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (BB10; Touch) AppleWebKit/537.10+ (KHTML, like Gecko) Version/10.0.9.2372 Mobile Safari/537.10+',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (PlayBook; U; RIM Tablet OS 2.1.0; en-US) AppleWebKit/536.2+ (KHTML like Gecko) Version/7.2.1.0 Safari/536.2+',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.3; en-us; SM-N900T Build/JSS15J) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.1; en-us; GT-N7100 Build/JRO03C) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.0; en-us; GT-I9300 Build/IMM76D) AppleWebKit/534.30 (KHTML, like Gecko) Version/4.0 Mobile Safari/534.30',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.0; SM-G950U Build/NRD90M) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/62.0.3202.84 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; SM-G965U Build/R16NW) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/63.0.3239.111 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.1.0; SM-T837A) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/70.0.3538.80 Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; en-us; KFAPWI Build/JDQ39) AppleWebKit/535.19 (KHTML, like Gecko) Silk/3.13 Safari/535.19 Silk-Accelerated=true',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; U; Android 4.4.2; en-us; LGMS323 Build/KOT49I.MS32310c) AppleWebKit/537.36 (KHTML, like Gecko) Version/4.0 Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Windows Phone 10.0; Android 4.2.1; Microsoft; Lumia 550) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/46.0.2486.0 Mobile Safari/537.36 Edge/14.14263',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Moto G (4)) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 10 Build/MOB31T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 4.4.2; Nexus 4 Build/KOT49H) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 5X Build/OPR4.170623.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 7.1.1; Nexus 6 Build/N6F26U) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Nexus 6P Build/OPP3.170518.006) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 6.0.1; Nexus 7 Build/MOB30X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (compatible; MSIE 10.0; Windows Phone 8.0; Trident/6.0; IEMobile/10.0; ARM; Touch; NOKIA; Lumia 520)',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (MeeGo; NokiaN9) AppleWebKit/534.13 (KHTML, like Gecko) NokiaBrowser/8.5.0 Mobile Safari/534.13',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 9; Pixel 3 Build/PQ1A.181105.017.A1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/66.0.3359.158 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 10; Pixel 4) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.138 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 11; Pixel 3) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/88.0.4324.181 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 5.0; SM-G900P Build/LRX21T) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0; Pixel 2 Build/OPD3.170816.012) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (Linux; Android 8.0.0; Pixel 2 XL Build/OPD1.170816.004) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Mobile Safari/537.36',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 10_3_1 like Mac OS X) AppleWebKit/603.1.30 (KHTML, like Gecko) Version/10.0 Mobile/14E304 Safari/602.1',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1',
'Connection': 'close'
    }, {
        'user-agent': 'Mozilla/5.0 (iPad; CPU OS 11_0 like Mac OS X) AppleWebKit/604.1.34 (KHTML, like Gecko) Version/11.0 Mobile/15A5341f Safari/604.1',
'Connection': 'close'
    }
]

headers = random.choice(headers_list)


# 将数字类型转化为字符串类型，并且将小于10的数字前添加0
def convert(x):
    x = int(x)
    if x < 10:
        x = f"0{x}"
    else:
        x = str(x)
    return x


# 计算下一天的日期
def day_to_nextday(y, m, d):
    date = datetime.datetime(y, m, d)
    next_day = date + datetime.timedelta(days=1)
    next_day = str(next_day).split(" ")[0]

    y1 = next_day.split("-")[0]
    m1 = next_day.split("-")[1]
    d1 = next_day.split("-")[2]

    y = convert(y)
    m = convert(m)
    d = convert(d)
    y1 = convert(y1)
    m1 = convert(m1)
    d1 = convert(d1)

    return y, m, d, y1, m1, d1


def down(y, m, d,north,west,east,south):
    y, m, d, y1, m1, d1 = day_to_nextday(y, m, d)  # 计算下一天日期
    sum = 0

    while True:
        try:
            url = f"https://ncss.hycom.org/thredds/ncss/GLBv0.08/expt_53.X?var=surf_el&var=water_temp_bottom&var=water_u_bottom&var=water_v_bottom&var=water_temp&var=water_u&var=water_v&north={north}&west={west}&east={east}&south={south}&disableProjSubset=on&horizStride=1&time_start={y}-{m}-{d}T00%3A00%3A00Z&time_end={y}-{m}-{d}T23%3A59%3A59Z&timeStride=1&vertCoord=&accept=netcdf"
            # 文件名   需修改
            file_Name = f"bbb/hycom_{y}{m}{d}.nc"
            print(f"{file_Name} downlaod...")
            headers = random.choice(headers_list)
            file = requests.get(url, headers=headers, timeout=360)
            open(file_Name, "wb").write(file.content)
            file_size = os.stat(file_Name).st_size
            if file_size > 1000:
                print(f"{file_Name} succ")
                break
            else:
                os.remove(file_Name)
                with open("error_reason.txt", "a") as f:
                    f.writelines(f"{file_Name} err\n")
            file.close()
        except Exception as e:
            print(f"{file_Name}  err   sum={sum}")
            print(f"Error reason: {e}")
            sum = sum + 1
            if sum > 10:
                with open("error_reason.txt", "a") as f:
                    f.writelines(f"{file_Name} err\n")
                sum = 0
                break


if __name__ == '__main__':
    # 起始时间  需修改
    year = 2010
    month = 8
    day = 13

    # 位置1
    # north = "20.60"
    # west = "120.10"
    # east = "120.30"
    # south = "20.40"


    # 位置2
    north = "17.60"
    west = "111.30"
    east = "111.50"
    south = "17.80"

    file_name = "error_reason.txt"
    if not os.path.exists(file_name):
        os.makedirs(file_name)

    while True:
        # 终止时间  需修改
        if (year == 2013) and (month == 10) and (day == 29):
            break

        try:
            down(year, month, day,north,west,east,south)  # 下载数据
        except Exception as e:
            year1 = convert(year)
            month1 = convert(month)
            day1 = convert(day)
            print(f"hycom_{year1}{month1}{day1}.nc err")
            print(f"Error reason: {e}")
            with open("error_reason.txt", "a") as f:
                f.writelines(f"hycom_{year1}{month1}{day1}.nc err\n")

        date1 = datetime.datetime(year, month, day)
        next_day1 = date1 + datetime.timedelta(days=1)
        next_day1 = str(next_day1).split(" ")[0]

        year = int(next_day1.split("-")[0])
        month = int(next_day1.split("-")[1])
        day = int(next_day1.split("-")[2])


