import glob
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import xarray as xr
import pandas as pd
import numpy as np
from config_4f import Config
import math
from datetime import datetime
import matplotlib.pyplot as plt

# 配置类
def config():
    folder_path = "/data3/hycom_surface_data"
    filenames = glob.glob(os.path.join(folder_path, "*nc"))
    filenames = sorted(filenames)

    return folder_path,filenames

# 根据位置和流速计算新的位置
def calculate_new_position(lat, lon, vn, ve, delta_t=21600):
    # 地球半径 (米)
    R = 6371000

    # 将当前的纬度和经度转换为弧度
    lat_rad = math.radians(lat)
    lon_rad = math.radians(lon)

    # 计算位移 (米)
    delta_y = vn * delta_t  # 经向位移 (南北)
    delta_x = ve * delta_t  # 纬向位移 (东西)

    # 计算新的纬度 (弧度)
    new_lat_rad = lat_rad + delta_y / R

    # 计算新的经度 (弧度)
    new_lon_rad = lon_rad + delta_x / (R * math.cos(lat_rad))

    # 将新的纬度和经度转换为度数
    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return new_lat, new_lon

# 为了方便查找对应的时间，只提取hycom月日
def parttime(filenames):
    part_filenames = []
    for i in range(len(filenames)):
        one = filenames[i].split(".")[0][-4:]
        part_filenames.append(one)
    return part_filenames

# 时间格式转换
def convert_time(sst_time):
    sst_time1 = []
    for j in range(len(sst_time)):
        one = str(sst_time[j]).split("T")
        two = one[0]
        three = one[-1].split(".")[0]
        four = f"{two} {three}"  # 格式为2019-07-11 06:00:00
        sst_time1.append(four)

    return sst_time1

# 用hycom流速计算下一个点
def hycom_newloc(last_lon_x,last_lat_x,last_time):
    folder_path,filenames = config()


    filename = str(last_time).split(" ")[0].split("-")
    filename = "".join(filename)
    filename = f"/hycom_surface_{filename}.nc"
    hycom_filename = f"{folder_path}{filename}"
    # print(hycom_filename)

    # 由于数据不全导致的问题，这里随便放一个文件进去就行，不全的数据会用气候态数据补全
    try:
        hycom_data = xr.open_dataset(hycom_filename)
    except:
        hycom_data = xr.open_dataset("/data3/hycom_surface_data/hycom_surface_20201027.nc")

    # 获取u和v
    sel_hycom_u = hycom_data["water_u"]
    sel_hycom_v = hycom_data["water_v"]

    # 获取位置信息、深度信息和时间信息，由于获取的hycom文件一样，这里取u作为标准
    sel_hycom_loc = sel_hycom_u.sel(lat=last_lat_x, lon=last_lon_x, method="nearest")
    sel_depth = sel_hycom_u["depth"].values  # 得到的是一个数组
    u = 0
    v = 0
    try:
        u = sel_hycom_u.loc[last_time, sel_depth[0], sel_hycom_loc.lat, sel_hycom_loc.lon].values
        v = sel_hycom_v.loc[last_time, sel_depth[0], sel_hycom_loc.lat, sel_hycom_loc.lon].values
    except:
        month_day = str(last_time).split(" ")[0].split("-")
        month_day1 = f"{month_day[1]}{month_day[2]}"  # 格式为0711这种
        month_day2 = f"{month_day[1]}-{month_day[2]}"  # 格式为07-11这种
        # 抽取时间 格式为06:00:00这种
        hours = str(last_time).split(" ")[-1]

        # 查找对应的hycom文件
        part_filenames = parttime(filenames)
        indeices = [j for j, name in enumerate(part_filenames) if
                    month_day1 in name]  # indeices最终为一个数组表明所查找的文件在part_fimenames中的第几个位置   part_fimenames和filenames的大小是一样的
        # 存储同一月日时的文件 这里不包含年是因为要靠年去区分
        one_datas = []
        us = []
        vs = []
        for indeic in indeices:  # 只是为了和indeics作区分，并没有写成index的形式
            name_file = filenames[indeic]
            hycom_data1 = xr.open_dataset(name_file)
            hycom_data_u1 = hycom_data1["water_u"]
            hycom_data_v1 = hycom_data1["water_v"]
            hycom_time1 = hycom_data1["time"].values  # 标准格式
            hycom_time2 = convert_time(hycom_time1)

            # 查找该文件是否存在这个时刻的数据
            one_1 = [j for j, name in enumerate(hycom_time2) if hours in name]
            one_2 = [j for j, name in enumerate(hycom_time2) if month_day2 in name]
            one = list(filter(lambda x: x in one_1, one_2))
            if one == []:  # 没有这个时刻的数据，跳过
                continue
            else:
                try:
                    uu = hycom_data_u1.loc[
                        hycom_time1[one[0]], sel_depth[0], sel_hycom_loc.lat, sel_hycom_loc.lon].values
                    vv = hycom_data_v1.loc[
                        hycom_time1[one[0]], sel_depth[0], sel_hycom_loc.lat, sel_hycom_loc.lon].values

                    us.append(uu)
                    vs.append(vv)
                except:
                    continue

        u = np.array(us).mean()
        v = np.array(vs).mean()

    new_lat, new_lon = calculate_new_position(last_lat_x, last_lon_x, v, u)
    return new_lat, new_lon


# 用hycom平均流速计算下一个点
def hycom_newloc_mean(last_lon_x,last_lat_x,last_time):
    folder_path, filenames = config()

    # 随便寻找一个分辨率是0.04且数据没有缺失的文件，找到标准的lat和lon,为了插值用
    std_filename = "/data3/hycom_surface_data/hycom_surface_20201027.nc"
    std_data = xr.open_dataset(std_filename)
    std_lat = std_data["lat"]
    std_lon = std_data["lon"]

    filename = str(last_time).split(" ")[0].split("-")
    filename = "".join(filename)
    filename = f"/hycom_surface_{filename}.nc"
    hycom_filename = f"{folder_path}{filename}"

    # 由于数据不全导致的问题，这里随便放一个文件进去就行，不全的数据会用气候态数据补全
    try:
        hycom_data = xr.open_dataset(hycom_filename)
    except:
        hycom_data = xr.open_dataset("/data3/hycom_surface_data/hycom_surface_20201027.nc")

    # 获取u
    sel_hycom_u = hycom_data["water_u"]

    # 获取v
    sel_hycom_v = hycom_data["water_v"]

    # 取的位置信息,深度信息,时间信息  由于文件一样，所以取u的作为标准
    # print(sel_hycom_sst["time"].values)
    sel_hycom_loc = sel_hycom_u.sel(lat=last_lat_x, lon=last_lon_x, method="nearest")
    # print(sel_hycom_loc)
    sel_depth = sel_hycom_u["depth"].values
    sel_time = sel_hycom_u["time"].values
    sel_time = convert_time(sel_time)  # 修改sel_time的格式

    # 划定范围
    rangepick = 0.5
    lat_0 = sel_hycom_loc.lat - rangepick
    lat_1 = sel_hycom_loc.lat + rangepick
    lon_0 = sel_hycom_loc.lon - rangepick
    lon_1 = sel_hycom_loc.lon + rangepick

    u = 0
    v = 0
    try:
        if sel_hycom_u["lat"].values.size == len(std_lat):  # 数据的分辨率和需要插值的分辨率相同
            a = sel_hycom_u.loc[last_time, sel_depth[0], lat_0:lat_1, lon_0:lon_1]

            a = pd.DataFrame(a.values)
            a.interpolate(method="linear", inplace=True)
            a.fillna(a.mean(), inplace=True)
            # a.fillna(method="bfill", inplace=True, axis=1)
            # a.fillna(method="ffill", inplace=True, axis=1)
            a.bfill(inplace=True, axis=1)
            a.ffill(inplace=True, axis=1)

            a = np.array(a).flatten()
        else:
            a = sel_hycom_u.loc[last_time, sel_depth[0]]
            a = a.interp(lon=std_lon, lat=std_lat)
            a = a.loc[lat_0:lat_1, lon_0:lon_1]

            a = pd.DataFrame(a.values)
            a.interpolate(method="linear", inplace=True)
            a.fillna(a.mean(), inplace=True)
            # a.fillna(method="bfill", inplace=True, axis=1)
            # a.fillna(method="ffill", inplace=True, axis=1)
            a.bfill(inplace=True, axis=1)
            a.ffill(inplace=True, axis=1)

            a = np.array(a).flatten()
    except:
        # 抽取月日
        month_day = last_time.split(" ")[0].split("-")
        month_day1 = f"{month_day[1]}{month_day[2]}"  # 格式为0711这种
        month_day2 = f"{month_day[1]}-{month_day[2]}"  # 格式为07-11这种
        # 抽取时间 格式为06:00:00这种
        hours = last_time.split(" ")[-1]

        # 查找对应的hycom文件
        part_filenames = parttime(filenames)
        indeices = [j for j, name in enumerate(part_filenames) if
                    month_day1 in name]  # indeices最终为一个数组表明所查找的文件在part_fimenames中的第几个位置   part_fimenames和filenames的大小是一样的

        # 存储同一月日时的文件 这里不包含年是因为要靠年去区分
        one_datas = []
        for indeic in indeices:  # 只是为了和indeics作区分，并没有写成index的形式
            name_file = filenames[indeic]
            # print(name_file)
            hycom_data1 = xr.open_dataset(name_file)
            hycom_data_sst1 = hycom_data1["water_u"]
            hycom_time1 = hycom_data1["time"].values  # 标准格式
            hycom_time2 = convert_time(hycom_time1)

            # 查找该文件是否存在这个时刻的数据
            one_1 = [j for j, name in enumerate(hycom_time2) if hours in name]
            one_2 = [j for j, name in enumerate(hycom_time2) if month_day2 in name]
            one = list(filter(lambda x: x in one_1, one_2))

            if one == []:  # 没有这个时刻的数据，跳过
                continue
            else:
                # 需要先向网格中进行插值然后再将数据append到one_datas中
                if hycom_data_sst1["lat"].values.size == len(std_lat):
                    aa = hycom_data_sst1.loc[hycom_time1[one[0]], sel_depth[0], lat_0:lat_1, lon_0:lon_1]

                    aa = pd.DataFrame(aa.values)
                    aa.interpolate(method="linear", inplace=True)
                    aa.fillna(aa.mean(), inplace=True)
                    # aa.fillna(method="bfill", inplace=True, axis=1)
                    # aa.fillna(method="ffill", inplace=True, axis=1)
                    aa.bfill(inplace=True,axis=1)
                    aa.ffill(inplace=True,axis=1)

                    aa = np.array(aa).flatten()
                else:
                    aa = hycom_data_sst1.loc[hycom_time1[one[0]], sel_depth[0]]
                    aa = aa.interp(lon=std_lon, lat=std_lat)
                    aa = aa.loc[lat_0:lat_1, lon_0:lon_1]

                    aa = pd.DataFrame(aa.values)
                    aa.interpolate(method="linear", inplace=True)
                    aa.fillna(aa.mean(), inplace=True)
                    # aa.fillna(method="bfill", inplace=True, axis=1)
                    # aa.fillna(method="ffill", inplace=True, axis=1)
                    aa.bfill(inplace=True,axis=1)
                    aa.ffill(inplace=True,axis=1)

                    aa = np.array(aa).flatten()
                one_datas.append(aa)

        one_datas = np.array(one_datas)
        one_datas_mean = one_datas.mean(axis=0)

        a = np.array(one_datas_mean).flatten()
    u = a.mean()


    try:
        if sel_hycom_v["lat"].values.size == len(std_lat):  # 数据的分辨率和需要插值的分辨率相同
            a = sel_hycom_v.loc[last_time, sel_depth[0], lat_0:lat_1, lon_0:lon_1]

            a = pd.DataFrame(a.values)
            a.interpolate(method="linear", inplace=True)
            a.fillna(a.mean(), inplace=True)
            # a.fillna(method="bfill", inplace=True, axis=1)
            # a.fillna(method="ffill", inplace=True, axis=1)
            a.bfill(inplace=True, axis=1)
            a.ffill(inplace=True, axis=1)

            a = np.array(a).flatten()
        else:
            a = sel_hycom_v.loc[last_time, sel_depth[0]]
            a = a.interp(lon=std_lon, lat=std_lat)
            a = a.loc[lat_0:lat_1, lon_0:lon_1]

            a = pd.DataFrame(a.values)
            a.interpolate(method="linear", inplace=True)
            a.fillna(a.mean(), inplace=True)
            # a.fillna(method="bfill", inplace=True, axis=1)
            # a.fillna(method="ffill", inplace=True, axis=1)
            a.bfill(inplace=True, axis=1)
            a.ffill(inplace=True, axis=1)

            a = np.array(a).flatten()
    except:
        # 抽取月日
        month_day = last_time.split(" ")[0].split("-")
        month_day1 = f"{month_day[1]}{month_day[2]}"  # 格式为0711这种
        month_day2 = f"{month_day[1]}-{month_day[2]}"  # 格式为07-11这种
        # 抽取时间 格式为06:00:00这种
        hours = last_time.split(" ")[-1]

        # 查找对应的hycom文件
        part_filenames = parttime(filenames)
        indeices = [j for j, name in enumerate(part_filenames) if
                    month_day1 in name]  # indeices最终为一个数组表明所查找的文件在part_fimenames中的第几个位置   part_fimenames和filenames的大小是一样的

        # 存储同一月日时的文件 这里不包含年是因为要靠年去区分
        one_datas = []
        for indeic in indeices:  # 只是为了和indeics作区分，并没有写成index的形式
            name_file = filenames[indeic]
            # print(name_file)
            hycom_data1 = xr.open_dataset(name_file)
            hycom_data_sst1 = hycom_data1["water_v"]
            hycom_time1 = hycom_data1["time"].values  # 标准格式
            hycom_time2 = convert_time(hycom_time1)

            # 查找该文件是否存在这个时刻的数据
            one_1 = [j for j, name in enumerate(hycom_time2) if hours in name]
            one_2 = [j for j, name in enumerate(hycom_time2) if month_day2 in name]
            one = list(filter(lambda x: x in one_1, one_2))

            if one == []:  # 没有这个时刻的数据，跳过
                continue
            else:
                # 需要先向网格中进行插值然后再将数据append到one_datas中
                if hycom_data_sst1["lat"].values.size == len(std_lat):
                    aa = hycom_data_sst1.loc[hycom_time1[one[0]], sel_depth[0], lat_0:lat_1, lon_0:lon_1]

                    aa = pd.DataFrame(aa.values)
                    aa.interpolate(method="linear", inplace=True)
                    aa.fillna(aa.mean(), inplace=True)
                    # aa.fillna(method="bfill", inplace=True, axis=1)
                    # aa.fillna(method="ffill", inplace=True, axis=1)
                    aa.bfill(inplace=True,axis=1)
                    aa.ffill(inplace=True,axis=1)

                    aa = np.array(aa).flatten()
                else:
                    aa = hycom_data_sst1.loc[hycom_time1[one[0]], sel_depth[0]]
                    aa = aa.interp(lon=std_lon, lat=std_lat)
                    aa = aa.loc[lat_0:lat_1, lon_0:lon_1]

                    aa = pd.DataFrame(aa.values)
                    aa.interpolate(method="linear", inplace=True)
                    aa.fillna(aa.mean(), inplace=True)
                    # aa.fillna(method="bfill", inplace=True, axis=1)
                    # aa.fillna(method="ffill", inplace=True, axis=1)
                    aa.bfill(inplace=True,axis=1)
                    aa.ffill(inplace=True,axis=1)

                    aa = np.array(aa).flatten()
                one_datas.append(aa)

        one_datas = np.array(one_datas)
        one_datas_mean = one_datas.mean(axis=0)

        a = np.array(one_datas_mean).flatten()
    v = a.mean()

    new_lat, new_lon = calculate_new_position(last_lat_x, last_lon_x, v, u)
    return new_lat, new_lon

# 计算经度差在该纬度上的距离
def longitude_distance_km(latitude, delta_longitude):
    # 地球周长（公里）
    earth_circumference_km = 40075.0

    # 将纬度转换为弧度
    latitude_radians = math.radians(latitude)

    # 计算经度差在该纬度上的距离
    distance_km = delta_longitude * math.cos(latitude_radians) * (earth_circumference_km / 360.0)

    distance_km = abs(distance_km)

    return distance_km


# 画baseline
def draw_baseline(ncname,length,trainpara):
    # ncname = "/home/ly/xia_datan/wangying/trace/trace/model_outputs/trackOnly_transformer/tensorboard/lightning_logs/version_2/checkpoints/tracecon_20240715122809.nc"
    dataset_1 = xr.open_dataset(ncname)

    folder_path = os.path.dirname(ncname)
    folder_path = f"{folder_path}/trace_pic"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)


    pred_result_lon = np.array(dataset_1["predictions_lon"])  # 预测lon(y)  984
    pred_result_lat = np.array(dataset_1["predictions_lat"])  # 预测lat(y)

    result_test_lon = np.array(dataset_1["test_lon"].values)  # 真实lon(y)  984
    result_test_lat = np.array(dataset_1["test_lat"].values)  # 真实lat(y)

    true_lon_y = np.array(dataset_1["true_lon_nodiff"].values)  # 真实lon偏差(y)  984
    true_lat_nodiff = np.array(dataset_1["true_lat_nodiff"].values)  # 真实lat偏差(y)

    true_lon_x = np.array(dataset_1["true_lon_nodiff_x"].values)  # 真实lon(x)   984
    true_lat_x = np.array(dataset_1["true_lat_nodiff_x"].values)  # 真实lat(x)

    trace_time = pd.read_csv(trainpara.time_id_path, header=None).values
    trace_time = trace_time[-length:]
    # print(len(trace_time))

    all_pred_sub_lon = []
    all_pred_sub_lat = []
    all_pres_sub_distance = []

    all_lon_sub_hycom = []
    all_lat_sub_hycom = []
    all_distance_sub_hycom = []

    all_lon_sub_hycom_mean = []
    all_lat_sub_hycom_mean = []
    all_diastance_sub_hycom_mean = []

    for i in range(length):
    # for i in range(1,10):
    #     print(i)
        if i%150 == 0:
            print(f"绘制进度{round((i*1.0/length*100),2)}%_____________________")
        last_lon_x = true_lon_x[i][-1]
        last_lat_x = true_lat_x[i][-1]
        last_time = trace_time[i][-10:]
        last_time = last_time[1::2]
        # print(last_time)

        # 预测轨迹
        pred_lon = pred_result_lon[i]
        pred_lon = np.concatenate(([last_lon_x],pred_lon))
        pred_lat = pred_result_lat[i]
        pred_lat = np.concatenate(([last_lat_x],pred_lat))

        # 真实轨迹
        result_lon = result_test_lon[i]
        result_lon = np.concatenate(([last_lon_x], result_lon))
        result_lat = result_test_lat[i]
        result_lat = np.concatenate(([last_lat_x], result_lat))



        pred_sub_lon = []
        pred_sub_distance = []
        for j in range(len(result_lon)):
            distance = longitude_distance_km(result_lon[j],abs(pred_lon[j]-result_lon[j]))
            pred_sub_lon.append(distance)
        pred_sub_lon = pred_sub_lon[1:]
        pred_sub_lat = abs((pred_lat - result_lat)[1:])
        pred_sub_lat = [x * (40075.0 / 360.0) for x in pred_sub_lat]
        for j in range(len(pred_sub_lon)):
            distance = math.sqrt(math.pow(pred_sub_lon[j],2) + math.pow(pred_sub_lat[j],2))
            pred_sub_distance.append(distance)
        all_pred_sub_lon.append(pred_sub_lon)
        all_pred_sub_lat.append(pred_sub_lat)
        all_pres_sub_distance.append(pred_sub_distance)




        lon_new_hycom = []
        lat_new_hycom = []
        lon_new_hycom.append(last_lon_x)
        lat_new_hycom.append(last_lat_x)
        new_lat = 0
        new_lon = 0
        for j in range(len(last_time) - 1):
            if j == 0:
                new_lat, new_lon = hycom_newloc(last_lon_x, last_lat_x, last_time[j])
                lon_new_hycom.append(new_lon)
                lat_new_hycom.append(new_lat)
            else:
                new_lat, new_lon = hycom_newloc(new_lon, new_lat, last_time[j])
                lon_new_hycom.append(new_lon)
                lat_new_hycom.append(new_lat)
        # print(lon_new_hycom)

        lon_sub_hycom = []
        for j in range(len(result_lon)):
            distance = longitude_distance_km(result_lon[j],abs(lon_new_hycom[j]-result_lon[j]))
            lon_sub_hycom.append(distance)
        lon_sub_hycom = lon_sub_hycom[1:]
        lat_sub_hycom = abs((lat_new_hycom - result_lat)[1:])
        lat_sub_hycom = [x * (40075.0 / 360.0) for x in lat_sub_hycom]
        distance_sub_hycom = []
        for j in range(len(lon_sub_hycom)):
            distance = math.sqrt(math.pow(lon_sub_hycom[j], 2) + math.pow(lat_sub_hycom[j], 2))
            distance_sub_hycom.append(distance)
        # print(distance_sub_hycom)
        all_lon_sub_hycom.append(lon_sub_hycom)
        all_lat_sub_hycom.append(lat_sub_hycom)
        all_distance_sub_hycom.append(distance_sub_hycom)


        lon_new_hycom_mean = []
        lat_new_hycom_mean = []
        lon_new_hycom_mean.append(last_lon_x)
        lat_new_hycom_mean.append(last_lat_x)
        new_lat_mean = 0
        new_lon_mean = 0
        for j in range(len(last_time) - 1):
            if j == 0:
                new_lat_mean, new_lon_mean = hycom_newloc_mean(last_lon_x, last_lat_x, last_time[j])
                lon_new_hycom_mean.append(new_lon_mean)
                lat_new_hycom_mean.append(new_lat_mean)
            else:
                new_lat_mean, new_lon_mean = hycom_newloc_mean(new_lon_mean, new_lat_mean, last_time[j])
                lon_new_hycom_mean.append(new_lon_mean)
                lat_new_hycom_mean.append(new_lat_mean)

        lon_sub_hycom_mean = []
        for j in range(len(result_lon)):
            distance = longitude_distance_km(result_lon[j], abs(lon_new_hycom_mean[j] - result_lon[j]))
            lon_sub_hycom_mean.append(distance)
        lon_sub_hycom_mean = lon_sub_hycom_mean[1:]
        lat_sub_hycom_mean = abs((lat_new_hycom_mean - result_lat)[1:])
        lat_sub_hycom_mean = [x * (40075.0 / 360.0) for x in lat_sub_hycom_mean]
        distance_sub_hycom_mean = []
        for j in range(len(lon_sub_hycom_mean)):
            distance = math.sqrt(math.pow(lon_sub_hycom_mean[j], 2) + math.pow(lat_sub_hycom_mean[j], 2))
            distance_sub_hycom_mean.append(distance)
        all_lon_sub_hycom_mean.append(lon_sub_hycom_mean)
        all_lat_sub_hycom_mean.append(lat_sub_hycom_mean)
        all_diastance_sub_hycom_mean.append(distance_sub_hycom_mean)

        if i%12 == 0:
            plt.plot(pred_lon, pred_lat, label="pred")
            plt.scatter(pred_lon, pred_lat)
            for j in range(len(pred_lon) - 1):
                plt.text(pred_lon[j + 1], pred_lat[j + 1], str(j + 13))

            plt.plot(result_lon, result_lat, label="true")
            plt.scatter(result_lon, result_lat)
            for j in range(len(result_lon) - 1):
                plt.text(result_lon[j + 1], result_lat[j + 1], str(j + 13))

            plt.plot(true_lon_x[i], true_lat_x[i], label="x")
            plt.scatter(true_lon_x[i], true_lat_x[i])
            for j in range(len(true_lon_x[i])):
                plt.text(true_lon_x[i][j], true_lat_x[i][j], str(j + 1))

            # plt.plot(lon_new_hycom, lat_new_hycom, label="hycom")
            # plt.scatter(lon_new_hycom, lat_new_hycom)
            #
            # plt.plot(lon_new_hycom_mean, lat_new_hycom_mean, label="hycom_mean")
            # plt.scatter(lon_new_hycom_mean, lat_new_hycom_mean)

            plt.legend()
            plt.title(f"{i}")
            plt.savefig(f"{folder_path}/{i}.png")
            plt.close()  # 清空画布
            # plt.show()

    data_all = []

    data_all.append(all_pred_sub_lon)
    data_all.append(all_pred_sub_lat)
    data_all.append(all_pres_sub_distance)

    data_all.append(all_lon_sub_hycom)
    data_all.append(all_lat_sub_hycom)
    data_all.append(all_distance_sub_hycom)

    data_all.append(all_lon_sub_hycom_mean)
    data_all.append(all_lat_sub_hycom_mean)
    data_all.append(all_diastance_sub_hycom_mean)
    return data_all


# 画箱图
def draw_box(all_pred_sub, all_sub_hycom, all_sub_hycom_mean, title,ncname):

    folder_path = os.path.dirname(ncname)
    folder_path = f"{folder_path}/boxplot"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    conditions = 4  # 时间点6,12,18,24
    models = 3  # pred、hycom、hycom_mean

    data = []
    data_1 = []

    for i in range(0, conditions):
        data_1.append([row[i] for row in all_pred_sub])
        data_1.append([row[i] for row in all_sub_hycom])
        data_1.append([row[i] for row in all_sub_hycom_mean])
        data.append(data_1)
        data_1 = []

    labels = [f'{(i + 1) * 6}' for i in range(conditions)]
    model_labels = ['pred', 'hycom', 'hycom_mean']
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    # 创建图表
    fig, ax = plt.subplots(figsize=(9, 6))

    # 计算每个组的位置
    positions = np.arange(conditions) * (models + 1)
    width = 0.8  # 每个箱线图的宽度
    for i in range(models):
        pos = positions + i
        bp = ax.boxplot([data[j][i] for j in range(conditions)], positions=pos, widths=width, patch_artist=True,
                        sym='None')
        for patch in bp['boxes']:
            patch.set_facecolor("none")
            patch.set_edgecolor(colors[i])

    # 添加中位数和平均值
    for i in range(conditions):
        for j in range(models):
            median = np.median(data[i][j])
            mean = np.mean(data[i][j])
            ax.plot(positions[i] + j, median, linestyle='-', color='r', linewidth=2)  # 中位数
            ax.plot(positions[i] + j, mean, 'bo')  # 平均值

    # 设置X轴标签和位置
    ax.set_xticks(positions + (models - 1) / 2)
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 设置图例
    handles = [plt.Line2D([0], [0], color=c, lw=4) for c in colors]
    handles += [plt.Line2D([0], [0], color='#ff7f0e', lw=2),
                plt.Line2D([0], [0], color='blue', marker='o', linestyle='None')]
    ax.legend(handles, model_labels + ['Median', 'Mean'])

    # 设置Y轴标签和标题
    ax.set_ylabel('Error Distance(km)')
    ax.set_xlabel('Forecast hour(h)')
    ax.set_title(title)

    # 显示图表
    # plt.tight_layout()
    plt.savefig(f"{folder_path}/{title}_boxplot.png")
    plt.close()
    # plt.show()


# 画误差图
def draw_error(data_all,ncname):
    folder_path = os.path.dirname(ncname)
    folder_path = f"{folder_path}/error"
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)

    all_pred_sub_lon = np.array(data_all[0])
    all_pred_sub_lat = np.array(data_all[1])
    all_pred_sub_distance = np.array(data_all[2])

    all_lon_sub_hycom = np.array(data_all[3])
    all_lat_sub_hycom = np.array(data_all[4])
    all_distance_sub_hycom = np.array(data_all[5])

    all_lon_sub_hycom_mean = np.array(data_all[6])
    all_lat_sub_hycom_mean = np.array(data_all[7])
    all_diastance_sub_hycom_mean = np.array(data_all[8])

    fig, axs = plt.subplots(4, 1, figsize=(6, 10))
    for i in range(0, 4):
        x = range(len(all_pred_sub_lon[:, 0]))
        pred = list(all_pred_sub_lon[:, i])
        hycom = list(all_lon_sub_hycom[:, i])
        hycom_mean = list(all_lon_sub_hycom_mean[:, i])

        axs[i].plot(x, pred, label="pred")
        axs[i].plot(x, hycom, label="hycom")
        axs[i].plot(x, hycom_mean, label="hycom_mean")

        axs[i].set_ylabel("Longitude error(km)")
        axs[i].text(-0.12, 0.5, f"{(i + 1) * 6}h", transform=axs[i].transAxes, fontsize=12, va='center', ha='right',
                    rotation=90)
        if i == 3:
            axs[i].set_xlabel("Test samples")
        if i < 3:
            axs[i].set_xticks([])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为图例腾出空间
    plt.savefig(f"{folder_path}/Longitude_error.png")
    plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(6, 10))
    for i in range(0, 4):
        x = range(len(all_pred_sub_lat[:, 0]))
        pred = list(all_pred_sub_lat[:, i])
        hycom = list(all_lat_sub_hycom[:, i])
        hycom_mean = list(all_lat_sub_hycom_mean[:, i])

        axs[i].plot(x, pred, label="pred")
        axs[i].plot(x, hycom, label="hycom")
        axs[i].plot(x, hycom_mean, label="hycom_mean")

        axs[i].set_ylabel("Latitude error(km)")
        axs[i].text(-0.12, 0.5, f"{(i + 1) * 6}h", transform=axs[i].transAxes, fontsize=12, va='center', ha='right',
                    rotation=90)
        if i == 3:
            axs[i].set_xlabel("Test samples")
        if i < 3:
            axs[i].set_xticks([])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为图例腾出空间
    plt.savefig(f"{folder_path}/Latitude_error.png")
    plt.close()

    fig, axs = plt.subplots(4, 1, figsize=(6, 10))
    for i in range(0, 4):
        x = range(len(all_pred_sub_distance[:, 0]))
        pred = list(all_pred_sub_distance[:, i])
        hycom = list(all_distance_sub_hycom[:, i])
        hycom_mean = list(all_diastance_sub_hycom_mean[:, i])

        axs[i].plot(x, pred, label="pred")
        axs[i].plot(x, hycom, label="hycom")
        axs[i].plot(x, hycom_mean, label="hycom_mean")

        axs[i].set_ylabel("Distance error(km)")
        axs[i].text(-0.12, 0.5, f"{(i + 1) * 6}h", transform=axs[i].transAxes, fontsize=12, va='center', ha='right',
                    rotation=90)
        if i == 3:
            axs[i].set_xlabel("Test samples")
        if i < 3:
            axs[i].set_xticks([])

    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.95), ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.93])  # 为图例腾出空间
    plt.savefig(f"{folder_path}/Distance_error.png")
    plt.close()


if __name__ == '__main__':
    trainpara = Config()
    ncname = "/home/ly/xia_datan/wangying/trace/trace_models_hycom_24_72/informer_4f_72_24/model_outputs_informer/tensorboard/lightning_logs/version_0/checkpoints/tracecon_20240823162705.nc"
    data_all = draw_baseline(ncname, 1967, trainpara)