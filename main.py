import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import akshare as ak
import warnings

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class RiskParityBacktest:
    """风险平价策略回测系统"""

    def __init__(self):
        self.data = None
        self.weights = None
        self.portfolio_values = None

    def get_asset_data(self):
        """获取资产数据"""
        print("获取资产数据...")

        # 使用更稳定的数据获取方式
        assets = {
            '沪深300': 'sh000300',  # 股票
            '中证500': 'sh000905',  # 股票
            '国债指数': 'sh000012',  # 债券 - 使用国债指数
            '黄金': 'AU9999'  # 黄金 - 使用黄金现货
        }

        all_data = []
        start_date = '2018-01-01'
        end_date = '2023-12-31'

        for asset_name, asset_code in assets.items():
            try:
                if asset_name in ['沪深300', '中证500', '国债指数']:
                    # 使用akshare的指数接口
                    df = ak.stock_zh_index_daily(symbol=asset_code)
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    df['asset'] = asset_name
                    df = df.rename(columns={'close': 'price'})
                    all_data.append(df[['date', 'asset', 'price']])
                    print(f"✓ 成功获取 {asset_name} 数据")

                elif asset_name == '黄金':
                    # 获取黄金数据
                    df = ak.stock_zh_index_daily(symbol='sh518880')  # 使用黄金ETF替代
                    df['date'] = pd.to_datetime(df['date'])
                    df = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
                    df['asset'] = asset_name
                    df = df.rename(columns={'close': 'price'})
                    all_data.append(df[['date', 'asset', 'price']])
                    print(f"✓ 成功获取 {asset_name} 数据")

            except Exception as e:
                print(f"✗ 获取 {asset_name} 数据失败: {e}")
                # 为失败的资产生成模拟数据
                simulated_data = self._generate_simulated_asset(asset_name, start_date, end_date)
                all_data.append(simulated_data)

        if not all_data:
            print("所有数据获取失败，使用完全模拟数据")
            return self._generate_all_simulated_data(start_date, end_date)

        # 合并所有数据
        self.data = pd.concat(all_data, ignore_index=True)

        # 确保日期格式一致
        self.data['date'] = pd.to_datetime(self.data['date'])

        # 创建宽表格式（每个资产一列）
        price_data = self.data.pivot(index='date', columns='asset', values='price')
        price_data = price_data.sort_index()  # 按日期排序

        # 前向填充缺失值
        price_data = price_data.ffill().bfill()

        self.data = price_data.reset_index()

        print(f"\n数据获取完成，时间范围: {self.data['date'].min()} 到 {self.data['date'].max()}")
        print(f"资产数量: {len(price_data.columns)}")
        return self.data

    def _generate_simulated_asset(self, asset_name, start_date, end_date):
        """生成模拟资产数据"""
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        np.random.seed(hash(asset_name) % 10000)

        # 不同资产有不同的波动特性
        if '300' in asset_name or '500' in asset_name:  # 股票
            base_price = 3000 if '300' in asset_name else 5000
            volatility = 0.015
            trend = 0.0003
        elif '国债' in asset_name:  # 债券
            base_price = 120
            volatility = 0.003
            trend = 0.0001
        else:  # 黄金
            base_price = 350
            volatility = 0.008
            trend = 0.0002

        prices = [base_price]
        for i in range(1, len(dates)):
            ret = np.random.normal(trend, volatility)
            # 添加一些市场相关性
            if i > 5 and prices[-1] > base_price * 1.2:
                ret -= 0.001  # 高位回调
            elif i > 5 and prices[-1] < base_price * 0.8:
                ret += 0.001  # 低位反弹

            new_price = prices[-1] * (1 + ret)
            prices.append(max(new_price, base_price * 0.5))  # 设置价格下限

        df = pd.DataFrame({
            'date': dates,
            'asset': asset_name,
            'price': prices
        })

        print(f"✓ 为 {asset_name} 生成模拟数据")
        return df

    def _generate_all_simulated_data(self, start_date, end_date):
        """生成所有资产的模拟数据"""
        assets = ['沪深300', '中证500', '国债指数', '黄金']
        all_data = []

        for asset_name in assets:
            df = self._generate_simulated_asset(asset_name, start_date, end_date)
            all_data.append(df)

        self.data = pd.concat(all_data, ignore_index=True)
        self.data['date'] = pd.to_datetime(self.data['date'])

        # 创建宽表
        price_data = self.data.pivot(index='date', columns='asset', values='price')
        price_data = price_data.sort_index()
        self.data = price_data.reset_index()

        return self.data

    def calculate_risk_parity_weights(self, lookback_days=252):
        """计算风险平价权重"""
        print("计算风险平价权重...")

        # 使用宽表数据
        price_data = self.data.set_index('date')

        # 计算收益率
        returns = price_data.pct_change().dropna()

        # 初始化权重DataFrame
        weights_df = pd.DataFrame(index=returns.index, columns=returns.columns)

        # 滚动计算权重
        for i in range(lookback_days, len(returns)):
            current_date = returns.index[i]

            # 获取过去lookback_days天的收益率数据
            lookback_returns = returns.iloc[i - lookback_days:i]

            # 计算波动率（年化）
            volatility = lookback_returns.std() * np.sqrt(252)

            # 风险平价权重：与波动率成反比
            risk_weights = 1 / volatility
            normalized_weights = risk_weights / risk_weights.sum()

            weights_df.loc[current_date] = normalized_weights

        self.weights = weights_df.dropna()
        print("权重计算完成")
        return self.weights

    def run_backtest(self, initial_capital=1000000):
        """运行回测"""
        print("开始回测...")

        # 使用宽表数据
        price_data = self.data.set_index('date')

        # 计算收益率
        returns = price_data.pct_change().dropna()

        # 确保权重和收益率的时间索引对齐
        common_dates = returns.index.intersection(self.weights.index)
        returns = returns.loc[common_dates]
        weights = self.weights.loc[common_dates]

        # 初始化投资组合价值
        portfolio_value = [initial_capital]
        benchmark_value = [initial_capital]  # 等权重基准
        dates = [returns.index[0] - pd.Timedelta(days=1)]  # 起始日期

        # 等权重组合作为基准
        equal_weights = pd.Series(1 / len(returns.columns), index=returns.columns)

        # 运行回测
        for i, (date, daily_returns) in enumerate(returns.iterrows()):
            if date in weights.index:
                current_weights = weights.loc[date]
            else:
                # 如果没有计算权重，使用等权重
                current_weights = equal_weights

            # 计算组合收益率
            portfolio_return = (current_weights * daily_returns).sum()
            portfolio_value.append(portfolio_value[-1] * (1 + portfolio_return))

            # 计算等权重基准收益率
            benchmark_return = (equal_weights * daily_returns).sum()
            benchmark_value.append(benchmark_value[-1] * (1 + benchmark_return))

            dates.append(date)

        # 创建结果DataFrame，确保使用DatetimeIndex
        self.portfolio_values = pd.DataFrame({
            'risk_parity': portfolio_value[1:],
            'equal_weight': benchmark_value[1:]
        }, index=pd.DatetimeIndex(dates[1:]))  # 使用DatetimeIndex

        print("回测完成")
        return self.portfolio_values

    def calculate_performance(self):
        """计算性能指标"""
        if self.portfolio_values is None:
            print("请先运行回测")
            return

        # 计算收益率
        risk_parity_returns = self.portfolio_values['risk_parity'].pct_change().dropna()
        equal_weight_returns = self.portfolio_values['equal_weight'].pct_change().dropna()

        # 年化收益率
        total_days = (self.portfolio_values.index[-1] - self.portfolio_values.index[0]).days
        total_years = total_days / 365.25

        risk_parity_total_return = (self.portfolio_values['risk_parity'].iloc[-1] /
                                    self.portfolio_values['risk_parity'].iloc[0]) - 1
        risk_parity_annual_return = (1 + risk_parity_total_return) ** (1 / total_years) - 1

        equal_weight_total_return = (self.portfolio_values['equal_weight'].iloc[-1] /
                                     self.portfolio_values['equal_weight'].iloc[0]) - 1
        equal_weight_annual_return = (1 + equal_weight_total_return) ** (1 / total_years) - 1

        # 年化波动率
        risk_parity_volatility = risk_parity_returns.std() * np.sqrt(252)
        equal_weight_volatility = equal_weight_returns.std() * np.sqrt(252)

        # 夏普比率（无风险利率3%）
        risk_free_rate = 0.03
        risk_parity_sharpe = (risk_parity_annual_return - risk_free_rate) / risk_parity_volatility
        equal_weight_sharpe = (equal_weight_annual_return - risk_free_rate) / equal_weight_volatility

        # 最大回撤
        risk_parity_drawdown = self._calculate_max_drawdown(self.portfolio_values['risk_parity'])
        equal_weight_drawdown = self._calculate_max_drawdown(self.portfolio_values['equal_weight'])

        performance = {
            '风险平价策略': {
                '年化收益率': risk_parity_annual_return,
                '年化波动率': risk_parity_volatility,
                '夏普比率': risk_parity_sharpe,
                '最大回撤': risk_parity_drawdown,
                '总收益率': risk_parity_total_return
            },
            '等权重基准': {
                '年化收益率': equal_weight_annual_return,
                '年化波动率': equal_weight_volatility,
                '夏普比率': equal_weight_sharpe,
                '最大回撤': equal_weight_drawdown,
                '总收益率': equal_weight_total_return
            }
        }

        print("\n" + "=" * 60)
        print("回测性能指标")
        print("=" * 60)

        for strategy, metrics in performance.items():
            print(f"\n{strategy}:")
            for metric, value in metrics.items():
                if '率' in metric or '收益' in metric:
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value:.4f}")

        return performance

    def _calculate_max_drawdown(self, values):
        """计算最大回撤"""
        peak = values.expanding().max()
        drawdown = (values - peak) / peak
        return drawdown.min()

    def plot_results(self):
        """绘制回测结果"""
        if self.portfolio_values is None:
            print("请先运行回测")
            return

        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 净值曲线
        risk_parity_nav = self.portfolio_values['risk_parity'] / self.portfolio_values['risk_parity'].iloc[0]
        equal_weight_nav = self.portfolio_values['equal_weight'] / self.portfolio_values['equal_weight'].iloc[0]

        axes[0, 0].plot(self.portfolio_values.index, risk_parity_nav,
                        label='风险平价策略', linewidth=2)
        axes[0, 0].plot(self.portfolio_values.index, equal_weight_nav,
                        label='等权重基准', linewidth=2, linestyle='--')
        axes[0, 0].set_title('策略净值曲线对比', fontsize=14, fontweight='bold')
        axes[0, 0].set_ylabel('累计收益')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 权重变化
        if self.weights is not None:
            # 每月抽样显示权重
            sample_weights = self.weights.resample('M').last()
            sample_weights.plot(ax=axes[0, 1], linewidth=2)
            axes[0, 1].set_title('风险平价权重变化', fontsize=14, fontweight='bold')
            axes[0, 1].set_ylabel('权重')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)

        # 3. 回撤分析
        risk_parity_drawdown = self._calculate_drawdown_series(self.portfolio_values['risk_parity'])
        equal_weight_drawdown = self._calculate_drawdown_series(self.portfolio_values['equal_weight'])

        axes[1, 0].fill_between(self.portfolio_values.index, risk_parity_drawdown * 100, 0,
                                alpha=0.3, label='风险平价策略')
        axes[1, 0].fill_between(self.portfolio_values.index, equal_weight_drawdown * 100, 0,
                                alpha=0.3, label='等权重基准')
        axes[1, 0].set_title('回撤分析', fontsize=14, fontweight='bold')
        axes[1, 0].set_ylabel('回撤 (%)')
        axes[1, 0].set_xlabel('日期')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 月度收益率分布
        risk_parity_monthly = self.portfolio_values['risk_parity'].resample('M').last().pct_change().dropna()
        equal_weight_monthly = self.portfolio_values['equal_weight'].resample('M').last().pct_change().dropna()

        axes[1, 1].hist(risk_parity_monthly * 100, bins=20, alpha=0.7,
                        label='风险平价策略', color='blue')
        axes[1, 1].hist(equal_weight_monthly * 100, bins=20, alpha=0.7,
                        label='等权重基准', color='orange')
        axes[1, 1].set_title('月度收益率分布', fontsize=14, fontweight='bold')
        axes[1, 1].set_xlabel('收益率 (%)')
        axes[1, 1].set_ylabel('频次')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('risk_parity_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()

        # 额外图表：年度表现
        self._plot_annual_performance()

    def _calculate_drawdown_series(self, values):
        """计算回撤序列"""
        peak = values.expanding().max()
        return (values - peak) / peak

    def _plot_annual_performance(self):
        """绘制年度表现图"""
        # 按年份分组计算收益率
        annual_returns = []

        for year in range(2018, 2024):
            year_data = self.portfolio_values[self.portfolio_values.index.year == year]
            if len(year_data) > 1:
                risk_parity_return = (year_data['risk_parity'].iloc[-1] / year_data['risk_parity'].iloc[0]) - 1
                equal_weight_return = (year_data['equal_weight'].iloc[-1] / year_data['equal_weight'].iloc[0]) - 1
                annual_returns.append({
                    '年份': year,
                    '风险平价策略': risk_parity_return,
                    '等权重基准': equal_weight_return
                })

        if not annual_returns:
            print("没有足够的年度数据")
            return

        annual_df = pd.DataFrame(annual_returns)

        plt.figure(figsize=(12, 6))
        x = np.arange(len(annual_df))
        width = 0.35

        plt.bar(x - width / 2, annual_df['风险平价策略'] * 100, width, label='风险平价策略', alpha=0.8)
        plt.bar(x + width / 2, annual_df['等权重基准'] * 100, width, label='等权重基准', alpha=0.8)

        plt.xlabel('年份')
        plt.ylabel('收益率 (%)')
        plt.title('年度收益率对比', fontsize=14, fontweight='bold')
        plt.xticks(x, annual_df['年份'])
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 在柱状图上添加数值
        for i, v in enumerate(annual_df['风险平价策略']):
            plt.text(i - width / 2, v * 100, f'{v:.1%}', ha='center', va='bottom' if v >= 0 else 'top')
        for i, v in enumerate(annual_df['等权重基准']):
            plt.text(i + width / 2, v * 100, f'{v:.1%}', ha='center', va='bottom' if v >= 0 else 'top')

        plt.tight_layout()
        plt.savefig('annual_performance.png', dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("风险平价资产配置策略回测系统")
    print("=" * 50)

    # 初始化回测系统
    backtest = RiskParityBacktest()

    # 获取数据
    backtest.get_asset_data()

    # 计算风险平价权重
    backtest.calculate_risk_parity_weights(lookback_days=252)

    # 运行回测
    backtest.run_backtest(initial_capital=1000000)

    # 计算性能指标
    performance = backtest.calculate_performance()

    # 绘制结果
    backtest.plot_results()

    print("\n回测完成！")
    print("结果图表已保存为PNG文件")


if __name__ == "__main__":
    main()