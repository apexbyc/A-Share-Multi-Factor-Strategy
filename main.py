import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import akshare as ak
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class PracticalMultiFactorBacktest:
    def __init__(self, transaction_cost=0.0015):
        self.data = None
        self.portfolio_values = None
        self.benchmark_returns = None
        self.transaction_cost = transaction_cost

    def get_current_constituents_data(self, start_date='20200101', end_date='20231231'):
        """
        实用的数据获取方法：使用当前成分股，但明确说明这个局限
        并在分析时考虑这个偏差的影响
        """
        print("获取当前沪深300成分股数据...")

        # 获取当前成分股
        try:
            constituent_df = ak.index_stock_cons_sina(symbol="000300")
            stock_codes = constituent_df['code'].tolist()
            print(f"获取到 {len(stock_codes)} 只当前成分股")
        except:
            # 备用方案
            stock_info = ak.stock_info_a_code_name()
            stock_codes = stock_info['code'].sample(100).tolist()
            print(f"使用备用股票池: {len(stock_codes)} 只股票")

        all_data = []
        success_count = 0

        for i, code in enumerate(stock_codes[:50]):  # 限制数量以便快速运行
            if i % 10 == 0:
                print(f"进度: {i}/50")

            try:
                # 获取价格数据
                stock_data = ak.stock_zh_a_hist(
                    symbol=code, period="daily",
                    start_date=start_date, end_date=end_date, adjust="hfq"
                )

                if stock_data.empty:
                    continue

                stock_data['代码'] = code
                stock_data['日期'] = pd.to_datetime(stock_data['日期'])
                stock_data = stock_data.rename(columns={
                    '日期': 'date', '收盘': 'close', '涨跌幅': 'pct_chg'
                })

                # 简化财务数据：使用模拟数据
                # 在实际项目中应该从数据库获取历史财务数据
                stock_data['PE'] = np.random.lognormal(2.5, 0.5)  # 模拟PE
                stock_data['ROE'] = np.random.normal(0.1, 0.05)  # 模拟ROE

                all_data.append(stock_data[['date', '代码', 'close', 'pct_chg', 'PE', 'ROE']])
                success_count += 1

            except Exception as e:
                continue

        if all_data:
            self.data = pd.concat(all_data, ignore_index=True)
            print(f"数据收集完成: {success_count} 只股票")
        else:
            print("使用模拟数据")
            self._generate_simulated_data()

        return self.data

    def _generate_simulated_data(self):
        """生成模拟数据用于演示"""
        print("生成模拟数据...")
        dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='D')
        stocks = [f'stock_{i:03d}' for i in range(50)]

        np.random.seed(42)
        all_data = []

        for stock in stocks:
            base_price = np.random.lognormal(3, 0.5) * 10
            prices = [base_price]

            # 生成有趋势的价格序列
            for i in range(1, len(dates)):
                # 加入市场相关性
                market_effect = np.random.normal(0.0002, 0.015)
                idiosyncratic = np.random.normal(0, 0.01)
                ret = market_effect + idiosyncratic
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.1))

            for i, date in enumerate(dates):
                all_data.append({
                    'date': date,
                    '代码': stock,
                    'close': prices[i],
                    'PE': np.random.lognormal(2.5, 0.3),
                    'ROE': np.random.normal(0.08, 0.03)
                })

        self.data = pd.DataFrame(all_data)

    def calculate_improved_factors(self):
        """改进的因子计算"""
        print("计算改进的因子...")

        self.data = self.data.sort_values(['代码', 'date']).reset_index(drop=True)

        # 计算20日动量
        self.data['momentum'] = self.data.groupby('代码')['close'].pct_change(20)

        # 使用MAD方法处理极端值
        def mad_outlier_detection(series, n=3):
            if series.notna().sum() < 10:
                return series
            median = series.median()
            mad = (series - median).abs().median()
            lower_bound = median - n * 1.4826 * mad
            upper_bound = median + n * 1.4826 * mad
            return np.clip(series, lower_bound, upper_bound)

        # 应用改进的极端值处理
        for factor in ['PE', 'ROE', 'momentum']:
            self.data[f'{factor}_clean'] = self.data.groupby('date')[factor].transform(mad_outlier_detection)

        return self.data

    def run_practical_backtest(self, initial_capital=1000000):
        """运行实用的回测"""
        print("开始回测...")

        # 获取月度调仓日期
        all_dates = sorted(self.data['date'].unique())
        rebalance_dates = []
        current_month = None
        for date in all_dates:
            if date.month != current_month:
                rebalance_dates.append(date)
                current_month = date.month

        print(f"回测期间: {rebalance_dates[0]} 到 {rebalance_dates[-1]}")
        print(f"调仓次数: {len(rebalance_dates)}")

        # 初始化投资组合
        portfolio = {
            'cash': initial_capital,
            'positions': {},
            'transaction_costs': 0
        }

        portfolio_history = []

        for i, rebalance_date in enumerate(rebalance_dates):
            if i % 6 == 0:
                print(f"处理 {rebalance_date.year}年{rebalance_date.month}月...")

            # 计算当前组合价值
            current_prices = self.data[self.data['date'] == rebalance_date].set_index('代码')['close']
            portfolio_value = portfolio['cash']
            for stock, shares in portfolio['positions'].items():
                if stock in current_prices.index:
                    portfolio_value += shares * current_prices[stock]

            # 记录组合价值
            portfolio_history.append({
                'date': rebalance_date,
                'value': portfolio_value,
                'cash': portfolio['cash'],
                'transaction_costs': portfolio['transaction_costs']
            })

            # 最后一次调仓不操作
            if i == len(rebalance_dates) - 1:
                break

            # 计算因子分数
            date_data = self._calculate_factor_scores(rebalance_date)
            if date_data is None or len(date_data) < 10:
                continue

            # 选取消分位股票
            try:
                date_data['quintile'] = pd.qcut(date_data['total_score'], 5, labels=False) + 1
                top_stocks = date_data[date_data['quintile'] == 5]
            except:
                continue

            if len(top_stocks) == 0:
                continue

            # 清空现有持仓（考虑交易成本）
            for stock, shares in portfolio['positions'].items():
                if stock in current_prices.index:
                    sell_value = shares * current_prices[stock]
                    cost = sell_value * self.transaction_cost
                    portfolio['cash'] += sell_value - cost
                    portfolio['transaction_costs'] += cost

            portfolio['positions'] = {}

            # 买入新持仓（考虑交易成本）
            stocks_to_buy = top_stocks['代码'].tolist()
            if stocks_to_buy:
                capital_per_stock = portfolio['cash'] / len(stocks_to_buy)

                for stock in stocks_to_buy:
                    price = current_prices.get(stock, 0)
                    if price > 0:
                        # 考虑买入成本
                        max_shares = int(capital_per_stock / (price * (1 + self.transaction_cost)))
                        if max_shares > 0:
                            buy_cost = max_shares * price * self.transaction_cost
                            portfolio['positions'][stock] = max_shares
                            portfolio['cash'] -= max_shares * price + buy_cost
                            portfolio['transaction_costs'] += buy_cost

        self.portfolio_values = pd.DataFrame(portfolio_history)

        # 计算基准
        self._calculate_benchmark(rebalance_dates)

        return self.portfolio_values

    def _calculate_factor_scores(self, date):
        """计算因子分数"""
        date_data = self.data[self.data['date'] == date].copy()
        if len(date_data) < 10:
            return None

        # 因子标准化打分
        for factor, col in [('PE', 'PE_clean'), ('ROE', 'ROE_clean'), ('momentum', 'momentum_clean')]:
            valid_data = date_data[date_data[col].notna()]
            if len(valid_data) > 0:
                if factor == 'PE':
                    date_data[f'{factor}_score'] = date_data[col].rank(ascending=True, pct=True)
                else:
                    date_data[f'{factor}_score'] = date_data[col].rank(ascending=False, pct=True)
            else:
                date_data[f'{factor}_score'] = 0.5

        # 等权合成（可以扩展为加权）
        date_data['total_score'] = (date_data['PE_score'] + date_data['ROE_score'] + date_data['momentum_score']) / 3

        return date_data

    def _calculate_benchmark(self, dates):
        """计算等权基准"""
        benchmark_values = [100]

        for i in range(1, len(dates)):
            start_date = dates[i - 1]
            end_date = dates[i]

            start_prices = self.data[self.data['date'] == start_date].set_index('代码')['close']
            end_prices = self.data[self.data['date'] == end_date].set_index('代码')['close']

            common_stocks = start_prices.index.intersection(end_prices.index)
            if len(common_stocks) == 0:
                benchmark_values.append(benchmark_values[-1])
                continue

            returns = []
            for stock in common_stocks:
                start_price = start_prices[stock]
                end_price = end_prices[stock]
                if start_price > 0:
                    ret = (end_price - start_price) / start_price
                    returns.append(ret)

            if returns:
                avg_return = np.mean(returns)
                benchmark_values.append(benchmark_values[-1] * (1 + avg_return))
            else:
                benchmark_values.append(benchmark_values[-1])

        self.benchmark_returns = pd.DataFrame({
            'date': dates,
            'value': benchmark_values
        })

    def analyze_performance(self):
        """分析性能"""
        if self.portfolio_values is None:
            print("请先运行回测")
            return

        portfolio_returns = self.portfolio_values['value'].pct_change().dropna()

        # 基础指标
        total_years = (self.portfolio_values['date'].iloc[-1] -
                       self.portfolio_values['date'].iloc[0]).days / 365.25
        total_return = (self.portfolio_values['value'].iloc[-1] /
                        self.portfolio_values['value'].iloc[0]) - 1
        annual_return = (1 + total_return) ** (1 / total_years) - 1

        # 风险调整收益
        volatility = portfolio_returns.std() * np.sqrt(252)
        sharpe_ratio = (annual_return - 0.03) / volatility if volatility > 0 else 0

        # 最大回撤
        cumulative = (1 + portfolio_returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        max_drawdown = drawdown.min()

        print("\n" + "=" * 50)
        print("回测结果分析")
        print("=" * 50)
        print(f"年化收益率: {annual_return:.2%}")
        print(f"年化波动率: {volatility:.2%}")
        print(f"夏普比率: {sharpe_ratio:.2f}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"累计交易成本: {self.portfolio_values['transaction_costs'].iloc[-1]:.2f} 元")

        return {
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }

    def plot_results(self):
        """绘制结果并保存"""
        if self.portfolio_values is None:
            print("请先运行回测")
            return

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        # 净值曲线
        portfolio_nav = self.portfolio_values['value'] / self.portfolio_values['value'].iloc[0]
        benchmark_nav = self.benchmark_returns['value'] / self.benchmark_returns['value'].iloc[0]

        ax1.plot(self.portfolio_values['date'], portfolio_nav, label='多因子策略', linewidth=2)
        ax1.plot(self.benchmark_returns['date'], benchmark_nav, label='等权基准', linewidth=2, linestyle='--')
        ax1.set_title('多因子策略 vs 基准', fontsize=14, fontweight='bold')
        ax1.set_ylabel('累计收益')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 收益率分布
        portfolio_returns = self.portfolio_values['value'].pct_change().dropna()
        ax2.hist(portfolio_returns * 100, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(portfolio_returns.mean() * 100, color='red', linestyle='--',
                    label=f'均值: {portfolio_returns.mean() * 100:.2f}%')
        ax2.set_title('月度收益率分布', fontsize=14, fontweight='bold')
        ax2.set_xlabel('收益率 (%)')
        ax2.set_ylabel('频次')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # 添加保存功能
        plt.savefig('strategy_performance.png', dpi=300, bbox_inches='tight')
        print("图表已保存为: strategy_performance.png")

        plt.show()


def quick_demo():
    """快速演示版本"""
    print("开始快速演示...")

    # 初始化
    backtest = PracticalMultiFactorBacktest(transaction_cost=0.0015)

    # 获取数据（使用当前成分股，但明确知道局限性）
    print("1. 数据获取...")
    backtest.get_current_constituents_data(start_date='20210101', end_date='20231231')

    # 计算因子
    print("2. 因子计算...")
    backtest.calculate_improved_factors()

    # 运行回测
    print("3. 回测执行...")
    backtest.run_practical_backtest(initial_capital=1000000)

    # 分析结果
    print("4. 结果分析...")
    results = backtest.analyze_performance()

    # 绘制图表
    print("5. 生成图表...")
    backtest.plot_results()

    print("\n演示完成!")
    print("注意: 此版本使用当前成分股回测历史，存在幸存者偏差")
    print("但在方法上展示了如何改进因子处理和交易成本考虑")

    return backtest, results


if __name__ == "__main__":
    # 直接运行这个快速演示版本
    backtest, results = quick_demo()
