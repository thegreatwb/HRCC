import collections

# 过程中需要用到的一些常量
kMinNumDeltas = 60
threshold_gain_ = 4
kBurstIntervalMs = 5
kTrendlineWindowSize = 20  # 用于求解趋势斜率的样本个数，每个样本为包组的单向延时梯度
kTrendlineSmoothingCoeff = 0.9
kOverUsingTimeThreshold = 10
kMaxAdaptOffsetMs = 15.0
eta = 1.08  # increasing coeffience for AIMDf
alpha = 0.85  # decreasing coeffience for AIMD
k_up_ = 0.0087
k_down_ = 0.039
Time_Interval = 200


class GCCEstimator(object):
    def __init__(self):
        # ----- 包组时间相关 -----
        self.packets_list = []  # 记录当前时间间隔内收到的所有包
        self.packet_group = []
        self.first_group_complete_time = -1  # 第一个包组的完成时间（该包组最后一个包的接收时间）

        # ----- 延迟相关/计算trendline相关 -----
        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])

        # ----- 预测带宽相关 -----
        self.state = 'Hold'
        self.last_bandwidth_estimation = 300 * 1000
        self.avg_max_bitrate_kbps_ = -1  # 最大码率的指数移动均值
        self.var_max_bitrate_kbps_ = -1  # 最大码率的方差
        self.rate_control_region_ = "kRcMaxUnknown"
        self.time_last_bitrate_change_ = -1  # 上一次变比特率的时间

        self.gamma1 = 12.5  # 检测过载的动态阈值
        self.num_of_deltas_ = 0  # delta的累计个数
        self.time_over_using = -1  # 记录over_using的时间
        self.prev_trend = 0.0  # 前一个trend
        self.overuse_counter = 0  # 对overuse状态计数
        self.overuse_flag = 'NORMAL'
        self.last_update_ms = -1  # 上一次更新阈值的时间
        self.last_update_threshold_ms = -1
        self.now_ms = -1  # 当前系统时间

        # with open("debug.log", 'w') as f:
        #     f.write("========================== debug.log =========================\n")
        # with open("bandwidth_estimated.txt", 'w') as f:
        #     f.write("========================== bandwidth_estimated.txt =========================\n")
        # with open("bandwidth_estimated_by_loss.txt", 'w') as f:
        #     f.write("========================== bandwidth_estimated_by_loss.txt =========================\n")
        # with open("bandwidth_estimated_by_delay.txt", 'w') as f:
        #     f.write("========================== bandwidth_estimated_by_delay.txt =========================\n")

    # add by wb: reset estimator according to rtc_env_gcc
    def reset(self):
        # ----- 包组时间相关 -----
        self.packets_list = []  # 记录当前时间间隔内收到的所有包
        self.packet_group = []
        self.first_group_complete_time = -1  # 第一个包组的完成时间（该包组最后一个包的接收时间）

        # ----- 延迟相关/计算trendline相关 -----
        self.acc_delay = 0
        self.smoothed_delay = 0
        self.acc_delay_list = collections.deque([])
        self.smoothed_delay_list = collections.deque([])

        # ----- 预测带宽相关 -----
        self.state = 'Hold'
        self.last_bandwidth_estimation = 300 * 1000
        self.avg_max_bitrate_kbps_ = -1  # 最大码率的指数移动均值
        self.var_max_bitrate_kbps_ = -1  # 最大码率的方差
        self.rate_control_region_ = "kRcMaxUnknown"
        self.time_last_bitrate_change_ = -1  # 上一次变比特率的时间

        self.gamma1 = 12.5  # 检测过载的动态阈值
        self.num_of_deltas_ = 0  # delta的累计个数
        self.time_over_using = -1  # 记录over_using的时间
        self.prev_trend = 0.0  # 前一个trend
        self.overuse_counter = 0  # 对overuse状态计数
        self.overuse_flag = 'NORMAL'
        self.last_update_ms = -1  # 上一次更新阈值的时间
        self.last_update_threshold_ms = -1
        self.now_ms = -1  # 当前系统时间

    def report_states(self, stats: dict):
        '''
        将200ms内接收到包的包头信息都存储于packets_list中
        :param stats: a dict with the following items
        :return: 存储200ms内所有包包头信息的packets_list
        '''
        pkt = stats
        packet_info = PacketInfo()
        packet_info.payload_type = pkt["payload_type"]
        packet_info.ssrc = pkt["ssrc"]
        packet_info.sequence_number = pkt["sequence_number"]
        packet_info.send_timestamp = pkt["send_time_ms"]
        packet_info.receive_timestamp = pkt["arrival_time_ms"]
        packet_info.padding_length = pkt["padding_length"]
        packet_info.header_length = pkt["header_length"]
        packet_info.payload_size = pkt["payload_size"]
        packet_info.size = pkt["header_length"] + pkt["payload_size"] + pkt["padding_length"]
        packet_info.bandwidth_prediction = self.last_bandwidth_estimation
        self.now_ms = packet_info.receive_timestamp  # 以最后一个包的到达时间作为系统时间

        # with open('debug.log', 'a+') as f:
        #     assert (isinstance(stats, dict))
        #     f.write(str(stats))
        #     f.write('\n')

        self.packets_list.append(packet_info)

    def get_estimated_bandwidth(self) -> int:
        '''
        计算估计带宽
        :return: 估计带宽 bandwidth_estimation
        '''
        # print("len(self.packets_list) = "+str(len(self.packets_list)))
        BWE_by_delay, flag = self.get_estimated_bandwidth_by_delay()
        # with open("bandwidth_estimated_by_delay.txt", 'a+') as f:
        #     bwe_delay = BWE_by_delay / 1000
        BWE_by_loss = self.get_estimated_bandwidth_by_loss()
        # with open("bandwidth_estimated_by_loss.txt", 'a+') as f:
        #     bwe_loss = BWE_by_loss / 1000
        bandwidth_estimation = min(BWE_by_delay, BWE_by_loss)
        if flag == True:
            self.packets_list = []  # 清空packets_list
        self.last_bandwidth_estimation = bandwidth_estimation
        return bandwidth_estimation,self.overuse_flag

    def get_inner_estimation(self):
        BWE_by_delay, flag = self.get_estimated_bandwidth_by_delay()
        # with open("bandwidth_estimated_by_delay.txt", 'a+') as f:
        #     bwe_delay = BWE_by_delay / 1000
        BWE_by_loss = self.get_estimated_bandwidth_by_loss()
        # with open("bandwidth_estimated_by_loss.txt", 'a+') as f:
        #     bwe_loss = BWE_by_loss / 1000
        bandwidth_estimation = min(BWE_by_delay, BWE_by_loss)
        if flag == True:
            self.packets_list = []  # 清空packets_list
        return BWE_by_delay,BWE_by_loss



    def change_bandwidth_estimation(self,bandwidth_prediction):
        self.last_bandwidth_estimation = bandwidth_prediction

    def get_estimated_bandwidth_by_delay(self):
        '''
        基于延迟的带宽预测
        :return: 基于延迟的估计带宽 bandwidth_estimation / 是否进行有效估计 flag
        '''
        if len(self.packets_list) == 0:  # 若该时间间隔内未收到包,则返回上一次带宽预测结果
            # print("len(self.packets_list) == 0")
            return self.last_bandwidth_estimation, False

        # 1. 分包组
        pkt_group_list = self.divide_packet_group()
        if len(pkt_group_list) < 2:  # 若仅有一个包组，返回上一次带宽预测结果
            return self.last_bandwidth_estimation, False

        # 2. 计算包组梯度
        send_time_delta_list, _, _, delay_gradient_list = self.compute_deltas_for_pkt_group(pkt_group_list)
        # with open('debug.log', 'a+') as f:
        #     f.write("delay_gradient_list = " + str(delay_gradient_list) + "\n")

        # 3. 计算斜率
        trendline = self.trendline_filter(delay_gradient_list, pkt_group_list)
        if trendline == None:  # 当窗口中样本数不够时，返回上一次带宽预测结果
            return self.last_bandwidth_estimation, False

        # 4. 判断当前网络状态
        # self.overuse_detector(trendline, send_time_delta_list[-1])
        self.overuse_detector(trendline, sum(send_time_delta_list))
        #print("current overuse_flag : " + str(self.overuse_flag))
        # 5. 给出带宽调整方向
        # state = self.state_transfer()
        state = self.ChangeState()
        #print("current state : " + str(state))
        # 6. 调整带宽
        bandwidth_estimation = self.rate_adaptation_by_delay(state)

        # with open("debug.log", 'a+') as f:
        #     bwe = bandwidth_estimation / 1000
        #     f.write("BWE by delay = " + str(int(bwe)) + " kbps" + ' ｜ ')
        return bandwidth_estimation, True

    def get_estimated_bandwidth_by_loss(self) -> int:
        '''
        基于丢包的带宽预测
        :return:基于丢包的估计带宽 bandwidth_estimation
        '''
        loss_rate = self.caculate_loss_rate()
        if loss_rate == -1:
            #print("len(self.packets_list) == 0")
            return self.last_bandwidth_estimation

        bandwidth_estimation = self.rate_adaptation_by_loss(loss_rate)

        # with open("debug.log", 'a+') as f:
        #     bwe = bandwidth_estimation / 1000
        #     f.write("BWE by loss = " + str(int(bwe)) + " kbps" + '\n')
        return bandwidth_estimation

    def caculate_loss_rate(self):
        '''
        计算该时段内的丢包率
        :return: 丢包率 loss_rate
        '''
        flag = False  # 标志是否获得第一个有效包
        valid_packets_num = 0
        min_sequence_number, max_sequence_number = 0, 0
        if len(self.packets_list) == 0:  # 该时间间隔内无包到达
            return -1
        for i in range(len(self.packets_list)):
            if self.packets_list[i].payload_type == 126:
                if not flag:
                    min_sequence_number = self.packets_list[i].sequence_number
                    max_sequence_number = self.packets_list[i].sequence_number
                    flag = True
                valid_packets_num += 1
                min_sequence_number = min(min_sequence_number, self.packets_list[i].sequence_number)
                max_sequence_number = max(max_sequence_number, self.packets_list[i].sequence_number)
        if (max_sequence_number - min_sequence_number) == 0:
            return -1
        receive_rate = valid_packets_num / (max_sequence_number - min_sequence_number)
        loss_rate = 1 - receive_rate
        return loss_rate

    def rate_adaptation_by_loss(self, loss_rate) -> int:
        '''
        根据丢包率计算估计带宽
        :param loss_rate: 丢包率
        :return: 基于丢包的预测带宽 bandwidth_estimation
        '''
        bandwidth_estimation = self.last_bandwidth_estimation
        if loss_rate > 0.1:
            bandwidth_estimation = self.last_bandwidth_estimation * (1 - 0.5 * loss_rate)
        elif loss_rate < 0.02:
            bandwidth_estimation = 1.05 * self.last_bandwidth_estimation
        return bandwidth_estimation

    def divide_packet_group(self):
        '''
        对接收到的包进行分组
        :return: 存有每个包组相关信息的pkt_group_list
        '''
        # todo:对乱序包和突发包的处理
        pkt_group_list = []
        first_send_time_in_group = self.packets_list[0].send_timestamp

        pkt_group = [self.packets_list[0]]
        for pkt in self.packets_list[1:]:
            if pkt.send_timestamp - first_send_time_in_group <= kBurstIntervalMs:
                pkt_group.append(pkt)
            else:
                pkt_group_list.append(PacketGroup(pkt_group))  # 填入前一个包组相关信息
                if self.first_group_complete_time == -1:
                    self.first_group_complete_time = pkt_group[-1].receive_timestamp
                first_send_time_in_group = pkt.send_timestamp
                pkt_group = [pkt]
        # pkt_group_list.append(PacketGroup(pkt_group))
        # with open('debug.log', 'a+') as f:
        #     f.write("num of groups = " + str(len(pkt_group_list)) + '\n')

        return pkt_group_list

    def compute_deltas_for_pkt_group(self, pkt_group_list):
        '''
        计算包组时间差
        :param pkt_group_list: 存有每个包组相关信息的list
        :return: 发送时间差、接收时间差、包组大小差、延迟梯度list
        '''
        send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list = [], [], [], []
        for idx in range(1, len(pkt_group_list)):  # 遍历每个包组
            send_time_delta = pkt_group_list[idx].send_time_list[-1] - pkt_group_list[idx - 1].send_time_list[-1]
            arrival_time_delta = pkt_group_list[idx].arrival_time_list[-1] - pkt_group_list[idx - 1].arrival_time_list[
                -1]
            group_size_delta = pkt_group_list[idx].pkt_group_size - pkt_group_list[idx - 1].pkt_group_size
            delay = arrival_time_delta - send_time_delta
            self.num_of_deltas_ += 1
            send_time_delta_list.append(send_time_delta)
            arrival_time_delta_list.append(arrival_time_delta)
            group_size_delta_list.append(group_size_delta)
            delay_gradient_list.append(delay)

        return send_time_delta_list, arrival_time_delta_list, group_size_delta_list, delay_gradient_list

    def trendline_filter(self, delay_gradient_list, pkt_group_list):
        '''
        根据包组的延时梯度计算斜率因子，判断延时变化的趋势
        :param delay_gradient_list: 延迟梯度list
        :param pkt_group_list: 存有每个包组信息的list
        :return: 趋势斜率trendline
        '''
        # print("delay_gradient_list : "+str(delay_gradient_list))
        for i, delay_gradient in enumerate(delay_gradient_list):
            accumulated_delay = self.acc_delay + delay_gradient
            smoothed_delay = kTrendlineSmoothingCoeff * self.smoothed_delay + (
                    1 - kTrendlineSmoothingCoeff) * accumulated_delay

            self.acc_delay = accumulated_delay
            self.smoothed_delay = smoothed_delay

            arrival_time_ms = pkt_group_list[i + 1].complete_time  # acc_delay_list
            self.acc_delay_list.append(arrival_time_ms - self.first_group_complete_time)

            self.smoothed_delay_list.append(smoothed_delay)  # smoothed_delay_list
            if len(self.acc_delay_list) > kTrendlineWindowSize:
                self.acc_delay_list.popleft()
                self.smoothed_delay_list.popleft()
        if len(self.acc_delay_list) == kTrendlineWindowSize:
            avg_acc_delay = sum(self.acc_delay_list) / len(self.acc_delay_list)
            avg_smoothed_delay = sum(self.smoothed_delay_list) / len(self.smoothed_delay_list)

            # 通过线性拟合求解延时梯度的变化趋势：
            numerator = 0
            denominator = 0
            for i in range(kTrendlineWindowSize):
                numerator += (self.acc_delay_list[i] - avg_acc_delay) * (
                        self.smoothed_delay_list[i] - avg_smoothed_delay)
                denominator += (self.acc_delay_list[i] - avg_acc_delay) * (self.acc_delay_list[i] - avg_acc_delay)

            # print("self.acc_delay_list : "+str(self.acc_delay_list))
            trendline = numerator / (denominator + 1e-05)
        else:
            trendline = None
            self.acc_delay_list.clear()
            self.smoothed_delay_list.clear()
            self.acc_delay = 0
            self.smoothed_delay = 0
        return trendline

    def overuse_detector(self, trendline, ts_delta):
        """
        根据滤波器计算的趋势斜率，判断当前是否处于过载状态
        :param trendline: 趋势斜率
        :param ts_delta: 发送时间间隔
        """
        # self.overuse_flag = 'NORMAL'
        now_ms = self.now_ms
        if self.num_of_deltas_ < 2:
            return

        modified_trend = trendline * min(self.num_of_deltas_, kMinNumDeltas) * threshold_gain_
        #print("modified_trend = " + str(modified_trend))

        # with open("debug.log", 'a+') as f:
        #     f.write("trendline = " + str(trendline) + " | prev_trendline = " + str(
        #         self.prev_trend) + '\n' + "modified_trend = " + str(modified_trend) + " | threshold = " + str(self.gamma1) + '\n')

        if modified_trend > self.gamma1:
            if self.time_over_using == -1:
                self.time_over_using = ts_delta / 2
            else:
                self.time_over_using += ts_delta
            self.overuse_counter += 1
            if self.time_over_using > kOverUsingTimeThreshold and self.overuse_counter > 1:
                if trendline > self.prev_trend:
                    self.time_over_using = 0
                    self.overuse_counter = 0
                    self.overuse_flag = 'OVERUSE'
        elif modified_trend < -self.gamma1:
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'UNDERUSE'
        else:
            self.time_over_using = -1
            self.overuse_counter = 0
            self.overuse_flag = 'NORMAL'

        self.prev_trend = trendline
        self.update_threthold(modified_trend, now_ms)  # 更新判断过载的阈值

        # with open("debug.log", 'a+') as f:
        #     f.write("overuse_flag = " + self.overuse_flag + '\n')

    def update_threthold(self, modified_trend, now_ms):
        '''
        更新判断过载的阈值
        :param modified_trend: 修正后的趋势
        :param now_ms: 当前系统时间
        :return: 无
        '''
        if self.last_update_threshold_ms == -1:
            self.last_update_threshold_ms = now_ms
        if abs(modified_trend) > self.gamma1 + kMaxAdaptOffsetMs:
            self.last_update_threshold_ms = now_ms
            return
        if abs(modified_trend) < self.gamma1:
            k = k_down_
        else:
            k = k_up_
        kMaxTimeDeltaMs = 100
        time_delta_ms = min(now_ms - self.last_update_threshold_ms, kMaxTimeDeltaMs)
        self.gamma1 += k * (abs(modified_trend) - self.gamma1) * time_delta_ms
        if (self.gamma1 < 6):
            self.gamma1 = 6
        elif (self.gamma1 > 600):
            self.gamma1 = 600
        self.last_update_threshold_ms = now_ms

    def state_transfer(self):
        '''
        更新发送码率调整方向
        :param overuse_flag: 网络状态
        :return: 新的调整方向
        '''
        newstate = None
        overuse_flag = self.overuse_flag
        if self.state == 'Decrease' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Decrease' and (overuse_flag == 'NORMAL' or overuse_flag == 'UNDERUSE'):
            newstate = 'Hold'
        elif self.state == 'Hold' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Hold' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Hold' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        elif self.state == 'Increase' and overuse_flag == 'OVERUSE':
            newstate = 'Decrease'
        elif self.state == 'Increase' and overuse_flag == 'NORMAL':
            newstate = 'Increase'
        elif self.state == 'Increase' and overuse_flag == 'UNDERUSE':
            newstate = 'Hold'
        else:
            print('Wrong state!')
        self.state = newstate
        return newstate

    def ChangeState(self):
        overuse_flag = self.overuse_flag
        if overuse_flag == 'NORMAL':
            if self.state == 'Hold':
                self.state = 'Increase'
        elif overuse_flag == 'OVERUSE':
            if self.state != 'Decrease':
                self.state = 'Decrease'
        elif overuse_flag == 'UNDERUSE':
            self.state = 'Hold'
        return self.state

    def rate_adaptation_by_delay(self, state):
        '''
        根据当前状态（hold, increase, decrease），决定最后的码率
        :param state: （hold, increase, decrease）
        :return: 估计码率
        '''

        # with open("debug.log",'a+') as f:
        #     f.write("state : "+str(state)+'\n')

        estimated_throughput = 0
        for pkt in self.packets_list:
            estimated_throughput += pkt.size
        if len(self.packets_list) == 0:
            estimated_throughput_bps = 0
        else:
            time_delta = self.now_ms - self.packets_list[0].receive_timestamp
            time_delta = max(time_delta , Time_Interval)
            estimated_throughput_bps = 1000 * 8 * estimated_throughput / time_delta
        estimated_throughput_kbps = estimated_throughput_bps / 1000
        # print("estimated_throughput_kbps = "+str(estimated_throughput_kbps))

        troughput_based_limit = 3 * estimated_throughput_bps + 10#todo disable this limitation to conbine with RL
        '''
        计算最大码率标准差
        最大码率标准差表征了链路容量 link capacity 的估计值相对于均值的波动程度。
        '''
        self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)
        std_max_bit_rate = pow(self.var_max_bitrate_kbps_ * self.avg_max_bitrate_kbps_, 0.5)

        if state == 'Increase':
            # 两个状态，在最大值附近，比最大值高到不知道哪里去
            if self.avg_max_bitrate_kbps_ >= 0 and \
                    estimated_throughput_kbps > self.avg_max_bitrate_kbps_ + 3 * std_max_bit_rate:
                self.avg_max_bitrate_kbps_ = -1.0
                self.rate_control_region_ = "kRcMaxUnknown"

            if self.rate_control_region_ == "kRcNearMax":
                # with open("debug.log",'a+') as f:
                #     f.write("rate_control_region_ == kRcNearMax\n")
                # 已经接近最大值了，此时增长需谨慎，加性增加
                additive_increase_bps = self.AdditiveRateIncrease(self.now_ms, self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + additive_increase_bps
            elif self.rate_control_region_ == "kRcMaxUnknown":
                # with open("debug.log",'a+') as f:
                #     f.write("rate_control_region_ == kRcMaxUnknown\n")
                multiplicative_increase_bps = self.MultiplicativeRateIncrease(self.now_ms,
                                                                              self.time_last_bitrate_change_)
                bandwidth_estimation = self.last_bandwidth_estimation + multiplicative_increase_bps
            else:
                print("error!")
            bandwidth_estimation = min(bandwidth_estimation,troughput_based_limit)
            self.time_last_bitrate_change_ = self.now_ms
        elif state == 'Decrease':
            # with open("debug.log", 'a+') as f:
            #     f.write("rate_control_region_ == "+str(self.rate_control_region_)+'\n')
            beta = 0.85
            bandwidth_estimation = beta * estimated_throughput_bps + 0.5
            if bandwidth_estimation > self.last_bandwidth_estimation:
                if self.rate_control_region_ != "kRcMaxUnknown":
                    bandwidth_estimation = (beta * self.avg_max_bitrate_kbps_ * 1000 + 0.5)
                bandwidth_estimation = min(bandwidth_estimation, self.last_bandwidth_estimation)
            self.rate_control_region_ = "kRcNearMax"

            if estimated_throughput_kbps < self.avg_max_bitrate_kbps_-3*std_max_bit_rate:
                # 当前速率小于均值较多，认为均值不可靠，复位
                self.avg_max_bitrate_kbps_ = -1
            # 衰减状态下需要更新最大均值
            self.UpdateMaxThroughputEstimate(estimated_throughput_kbps)

            # 降低码率后回到HOLD状态，如果网络状态仍然不好，在Overuse仍然会进入Dec状态。
            # 如果恢复，则不会是Overuse，会保持或增长。
            self.state='Hold'
            self.time_last_bitrate_change_ = self.now_ms
        elif state == 'Hold':
            bandwidth_estimation = self.last_bandwidth_estimation
        else:
            print('Wrong State!')
        return bandwidth_estimation

    def AdditiveRateIncrease(self, now_ms, last_ms):
        """
        执行加性码率增长算法
        """
        # 计算平均包大小
        sum_packet_size, avg_packet_size = 0, 0
        for pkt in self.packets_list:
            sum_packet_size += pkt.size
        avg_packet_size = 8 * sum_packet_size / len(self.packets_list)

        beta = 0.0
        RTT = 2 * (self.packets_list[-1].receive_timestamp - self.packets_list[-1].send_timestamp)
        response_time = 200

        if last_ms > 0:
            beta = min(((now_ms - last_ms) / response_time), 1.0)
        additive_increase_bps = max(800, beta * avg_packet_size)
        return additive_increase_bps

    def MultiplicativeRateIncrease(self, now_ms, last_ms):
        """
        执行乘性增大算法
        :param now_ms: 当前时间
        :param last_ms: 上次码率被更新的时间
        :return: 增加的码率大小
        """
        alpha = 1.08
        if last_ms > -1:
            time_since_last_update_ms = min(now_ms - last_ms, 1000)
            alpha = pow(alpha, time_since_last_update_ms / 1000)
        multiplicative_increase_bps = max(self.last_bandwidth_estimation * (alpha - 1.0), 1000.0)
        return multiplicative_increase_bps

    def UpdateMaxThroughputEstimate(self, estimated_throughput_kbps):
        """
        输入网络带宽过载状态下的码率估计值 estimated_throughput_kbps，计算网络链路容量的方差。
        """
        # 一次指数平滑法计算最大码率的指数移动均值
        alpha = 0.05
        if self.avg_max_bitrate_kbps_ == -1:
            self.avg_max_bitrate_kbps_ = estimated_throughput_kbps
        else:
            self.avg_max_bitrate_kbps_ = (1 - alpha) * self.avg_max_bitrate_kbps_ + alpha * estimated_throughput_kbps
        # 一次指数平滑法计算最大码率的方差
        norm = max(self.avg_max_bitrate_kbps_, 1.0)
        var_value = pow((self.avg_max_bitrate_kbps_ - estimated_throughput_kbps), 2) / norm  # 归一化
        self.var_max_bitrate_kbps_ = (1 - alpha) * self.var_max_bitrate_kbps_ + alpha * var_value
        # 将归一化后的方差控制到 [0.4, 2.5] 区间范围之内
        if self.var_max_bitrate_kbps_ < 0.4:
            self.var_max_bitrate_kbps_ = 0.4
        if self.var_max_bitrate_kbps_ > 2.5:
            self.var_max_bitrate_kbps_ = 2.5


class PacketInfo:
    def __init__(self):
        self.payload_type = None
        self.sequence_number = None  # int
        self.send_timestamp = None  # int, ms
        self.ssrc = None  # int
        self.padding_length = None  # int, B
        self.header_length = None  # int, B
        self.receive_timestamp = None  # int, ms
        self.payload_size = None  # int, B
        self.bandwidth_prediction = None  # int, bps


# 定义包组的类，记录一个包组的相关信息
class PacketGroup:
    def __init__(self, pkt_group):
        self.pkts = pkt_group
        self.arrival_time_list = [pkt.receive_timestamp for pkt in pkt_group]
        self.send_time_list = [pkt.send_timestamp for pkt in pkt_group]
        self.pkt_group_size = sum([pkt.size for pkt in pkt_group])
        self.pkt_num_in_group = len(pkt_group)
        self.complete_time = self.arrival_time_list[-1]
        self.transfer_duration = self.arrival_time_list[-1] - self.arrival_time_list[0]
