import math
import matplotlib.pyplot as plt
from enum import Enum
from typing import Dict, List, Any, Tuple

class METRIC_TYPES(Enum):
    ATE = 'ate'

class PerformanceMetrics:
    # Calculate pose between two poses
    @staticmethod
    def calc_distance(ytrue: List[int], ymeas: List[int]) -> float:
        #print(ytrue, ymeas)
        sum_of_squares = math.pow(ymeas[0] - ytrue[0], 2) + \
                         math.pow(ymeas[1] - ytrue[1], 2)
        dist = math.sqrt(math.fabs(sum_of_squares))
        return dist
        
    # Calculate Accuracy
    @staticmethod
    def calc_acc(true_val: float, meas_val: float) -> Any:
        try:
            return 100 - PerformanceMetrics.calc_err_rate(true_val, meas_val)
        except:
            return None
    
    # Calculate Error Rate
    @staticmethod
    def calc_err_rate(true_val: float, meas_val: float) -> Any:
        try:
            return 100*(math.fabs(meas_val - true_val)/true_val)
        except:
            return None
    
    @staticmethod
    def get_index_by_time(timestamp: str, start_idx: int, poses: List[int]) -> int:
        #print("Length:", len(poses))
        for i in range(start_idx, len(poses)):
            if int(poses[i]) == int(timestamp):
                #print(f"Stamp: {timestamp}, Pose: {poses[i]}")
                return i
        return -1
    
    @staticmethod
    def get_point(data: List[List[int]], idx: int) -> List[int]:
        tmp = []
        #print(data)
        for i in range(1, len(data)):
            tmp.append(data[i][idx])
        return tmp
    
    @staticmethod
    def show_table(title: str, filename: str, rows: List[str], cols: List[str], collection: List[Any], order: List[Any], path: str, save_file: bool) -> None:
        import os
        print("Preparing performance metrics...")
        fig, axs = plt.subplots(len(collection),1)
        fig.patch.set_visible(False) # type: ignore

        for i in range(len(collection)):
            axs[i].axis('off')
            axs[i].axis('tight')
            table = axs[i].table(cellText=collection[i], colLabels=cols, rowLabels=rows, loc='center')
            table.scale(1, 1.5)
            axs[i].set_title(order[i], loc='left')

        plt.suptitle(title, size= 15)
        # plt.figtext(0.95, 0.05, timestamp, horizontalalignment='right', size=10, weight='light')
        fig.tight_layout()
        if save_file == True:
            filepath = os.path.join(path, filename)
            plt.savefig(filepath)
        else:
            plt.show()

###################################
# Absolute Trajectory Error (ATE) #
################################### 

class ATE:
    # RPE Root Square Mean Error (Pose Difference)
    @staticmethod
    def calc_ate_rmse(ytrue: List[List[int]], ymeas: List[List[Any]]) -> Any:
        try:
            sum = 0
            count = 0
            try:
                for i in range(len(ymeas[0])):
                    j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                    ytrue_pt = PerformanceMetrics.get_point(ytrue, j)
                    ymeas_pt = PerformanceMetrics.get_point(ymeas, i)
                    sum += math.pow(PerformanceMetrics.calc_distance(ytrue_pt, ymeas_pt), 2)
                    count += 1
            except Exception as e:
                print(e)
            sum /= count
            #print("RSME", round(math.sqrt(sum), 4))
            return round(math.sqrt(sum), 4)
        except Exception as e:
            return None
        
    # RPE Mean (Pose Difference)
    @staticmethod
    def calc_ate_mean(ytrue: List[List[int]], ymeas: List[List[Any]]) -> float:
        mu = 0
        count = 0
        try:
            for i in range(len(ymeas)):
                j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                ytrue_pt = PerformanceMetrics.get_point(ytrue, j)
                ymeas_pt = PerformanceMetrics.get_point(ymeas, i)
                mu += PerformanceMetrics.calc_distance(ytrue_pt, ymeas_pt)
                count +=1
            mu /= count
        except Exception as e:
            print(e)
        #print("Mean: ", round(mu, 4))
        return round(mu, 4)

    # RPE Standard Deviation (Pose Difference)
    @staticmethod
    def calc_ate_sd(mu: float, ytrue: List[List[int]], ymeas: List[List[Any]]) -> float:
        std = 0
        count = 0
        try:
            for i in range(len(ymeas)):
                j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                ytrue_pt = PerformanceMetrics.get_point(ytrue, j)
                ymeas_pt = PerformanceMetrics.get_point(ymeas, i)
                dist = PerformanceMetrics.calc_distance(ytrue_pt, ymeas_pt)
                std += math.pow((dist - mu), 2)
                count += 1
            std /= count
        except Exception as e:
            print(e)
        #print("SD: ", round(math.sqrt(std), 4))
        return round(math.sqrt(std), 4)
    
    @staticmethod
    def get_pose_diff_metrics(ytrue: List[List[int]], ymeas: List[List[Any]], title: str):
        rmse = ATE.calc_ate_rmse(ytrue, ymeas)
        mu = ATE.calc_ate_mean(ytrue, ymeas)
        sd = ATE.calc_ate_sd(mu, ytrue, ymeas)
        return (rmse, mu, sd)
    
    # ATE Root Square Mean Error (Single Value)
    @staticmethod
    def calc_ate_rmse2(ytrue: List[List[int]], ymeas: List[List[Any]], val_type: str) -> float:
        sum = 0
        count = 0
        try:
            c_idx = -1
            for i in range(len(ymeas)):
                j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                if val_type == "x":
                    c_idx = 1
                elif val_type == "y":
                    c_idx = 2
                elif val_type == "z":
                    c_idx = 3
                sum += math.pow(math.fabs(ytrue[c_idx][j] - ymeas[c_idx][i]), 2)
                count += 1
            sum /= count
        except Exception as e:
            print(e)
        #print("RMSE: ", round(math.sqrt(sum), 2))
        return round(math.sqrt(sum), 2)
        
    # ATE Mean (Single Value)
    @staticmethod
    def calc_ate_mean2(ytrue: List[List[int]], ymeas: List[List[Any]] ,val_type: str) -> float:
        mu = 0
        count = 0
        try:
            c_idx = -1
            for i in range(len(ymeas)):
                j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                if val_type == "x":
                    c_idx = 1
                elif val_type == "y":
                    c_idx = 2
                elif val_type == "z":
                    c_idx = 3
                mu += math.fabs(ytrue[c_idx][j] - ymeas[c_idx][i])
                count +=1
            mu /= count
        except Exception as e:
            print(e)
        #print("Mean: ", round(mu, 4))
        return round(mu, 4)

    # ATE Standard Deviation (Single Value) 
    @staticmethod   
    def calc_ate_sd2(mu: float, ytrue:List[List[int]], ymeas: List[List[Any]], val_type: str) -> float:
        std = 0
        count = 0
        try:
            c_idx = -1
            for i in range(len(ymeas)):
                j = PerformanceMetrics.get_index_by_time(ymeas[0][i], i, ytrue[0])
                if val_type == "x":
                    c_idx = 1
                elif val_type == "y":
                    c_idx = 2
                elif val_type == "z":
                    c_idx = 3
                std += math.pow((ytrue[c_idx][j] -ymeas[c_idx][i]) - mu, 2)
                count += 1
            std /= count
        except Exception as e:
            print(e)
        #print("SD: ", round(math.sqrt(std), 4))
        return round(math.sqrt(std), 4)

    @staticmethod
    def get_single_val_metrics(ytrue: List[List[int]], ymeas: List[List[int]], val_type: str) -> Tuple[float,...]:
        rmse = ATE.calc_ate_rmse2(ytrue, ymeas, val_type)
        mu = ATE.calc_ate_mean2(ytrue, ymeas, val_type)
        sd = ATE.calc_ate_sd2(mu, ytrue, ymeas, val_type)
        return (rmse, mu, sd)