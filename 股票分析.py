import streamlit as st
import pandas as pd
import tushare as ts
import time
import numpy as np

def check_streamlit_run():
    """检查是否使用 streamlit run 命令运行"""
    try:
        st.session_state
    except:
        print("\n错误：请使用以下命令运行此应用：")
        print("\nstreamlit run your_file_address.py\n")
        exit(1)

# 初始化 Session State
if not hasattr(st, 'session_state'):
    check_streamlit_run()

if 'stock_data' not in st.session_state:
    st.session_state['stock_data'] = pd.DataFrame()
    st.session_state['last_updated'] = None

# 配置 Tushare Token
TUSHARE_TOKEN = 'bee2a7f571323f0bb8036a09e6982ff65d66c7e334565211c3825398'

def validate_token():
    """验证 Tushare Token 是否有效"""
    try:
        ts.set_token(TUSHARE_TOKEN)
        pro = ts.pro_api()
        # 尝试进行一个简单的API调用来验证token
        pro.query('stock_basic', limit=1)
        return True, pro
    except Exception as e:
        st.error(f"Tushare Token 验证失败: {str(e)}")
        st.error("请确保您的 Token 有效。访问 https://tushare.pro/register?reg=407338 注册或更新 Token")
        return False, None

def get_latest_trade_date(pro):
    """获取最近的交易日期"""
    try:
        today = time.strftime('%Y%m%d')
        trade_cal = pro.trade_cal(
            exchange='SSE',
            start_date='20250101',
            end_date=today,
            is_open='1'
        )
        if trade_cal.empty:
            return None
        return trade_cal['cal_date'].iloc[-1]
    except Exception as e:
        st.error(f"获取交易日历失败: {str(e)}")
        return None

def safe_numeric_conversion(df, column, scale_factor=None):
    """安全地将列转换为数值类型"""
    try:
        # 首先替换特殊值
        df[column] = df[column].replace([np.inf, -np.inf], np.nan)
        
        # 转换为数值类型
        df[column] = pd.to_numeric(df[column], errors='coerce')
        
        # 应用缩放因子（如果提供）
        if scale_factor is not None:
            df[column] = df[column] * scale_factor
        
        # 替换无效值为0
        df[column] = df[column].fillna(0)
        
        return df
    except Exception as e:
        st.warning(f"处理 {column} 字段时出现警告: {str(e)}")
        df[column] = 0
        return df

def fetch_stock_data():
    """使用 Tushare API 获取 A 股行情数据"""
    try:
        # 验证 Token
        is_valid, pro = validate_token()
        if not is_valid:
            return pd.DataFrame()
        
        # 获取最近交易日
        latest_trade_date = get_latest_trade_date(pro)
        if not latest_trade_date:
            st.error("无法获取最近交易日数据")
            return pd.DataFrame()
        
        st.info(f"正在获取 {latest_trade_date} 的交易数据...")
        
        try:
            # 首先获取基本信息
            df_basic = pro.stock_basic(
                exchange='',
                list_status='L',
                fields='ts_code,name'
            )
            
            # 获取当日行情数据
            df_daily = pro.daily(
                trade_date=latest_trade_date,
                fields='ts_code,open,high,low,close,pre_close,change,pct_chg,vol,amount'
            )
            
            # 获取每日指标数据
            df_daily_basic = pro.daily_basic(
                trade_date=latest_trade_date,
                fields='ts_code,pe_ttm,pb'
            )
            
            if df_daily.empty:
                st.error("未获取到行情数据")
                return pd.DataFrame()
            
            # 合并数据
            df = pd.merge(df_basic, df_daily, on='ts_code', how='inner')
            df = pd.merge(df, df_daily_basic, on='ts_code', how='left')
            
            # 重命名列
            df = df.rename(columns={
                'ts_code': '股票代码',
                'name': '股票名称',
                'close': '最新价',
                'pct_chg': '涨跌幅',
                'vol': '成交量',
                'amount': '成交额',
                'pe_ttm': '滚动市盈率',
                'pb': '市净率',
                'open': '开盘价',
                'high': '最高价',
                'low': '最低价',
                'pre_close': '昨收价',
                'change': '涨跌额'
            })
            
            # 确保所有必要的列都存在
            required_columns = ['股票代码', '股票名称', '最新价', '涨跌幅', '成交量', '成交额', '滚动市盈率', '市净率']
            for col in required_columns:
                if col not in df.columns:
                    st.warning(f"缺少 {col} 列，将使用默认值")
                    df[col] = 0 if col not in ['股票代码', '股票名称'] else ''
            
            # 数据类型转换和单位调整
            numeric_columns = {
                '最新价': 1,
                '开盘价': 1,
                '最高价': 1,
                '最低价': 1,
                '昨收价': 1,
                '涨跌额': 1,
                '涨跌幅': 1,
                '成交量': 1,  # 单位：手
                '成交额': 0.0001,  # 转换为万元
                '滚动市盈率': 1,
                '市净率': 1
            }
            
            for col, scale in numeric_columns.items():
                if col in df.columns:
                    try:
                        df[col] = pd.to_numeric(df[col], errors='coerce') * scale
                    except Exception as e:
                        st.warning(f"转换 {col} 列时出错: {str(e)}")
                        df[col] = 0
            
            # 处理缺失值
            df = df.fillna({
                '最新价': 0,
                '开盘价': 0,
                '最高价': 0,
                '最低价': 0,
                '昨收价': 0,
                '涨跌额': 0,
                '涨跌幅': 0,
                '成交量': 0,
                '成交额': 0,
                '滚动市盈率': 0,
                '市净率': 0
            })
            
            # 按涨跌幅排序
            df = df.sort_values(by='涨跌幅', ascending=False)
            
            st.success(f"成功获取 {len(df)} 条股票数据")
            return df
        
        except Exception as e:
            st.error(f"获取股票数据失败: {str(e)}")
            return pd.DataFrame()
    
    except Exception as e:
        st.error(f"获取数据失败: {str(e)}")
        st.error("如果问题持续存在，请检查网络连接或联系技术支持")
        return pd.DataFrame()

def create_price_distribution(df):
    """创建价格分布统计"""
    try:
        # 创建价格区间
        bins = 10
        price_range = np.linspace(df['最新价'].min(), df['最新价'].max(), bins + 1)
        labels = [f'{price_range[i]:.2f}-{price_range[i+1]:.2f}' for i in range(bins)]
        
        # 统计每个区间的股票数量
        df['价格区间'] = pd.cut(df['最新价'], bins=bins, labels=labels)
        price_dist = df.groupby('价格区间').size().reset_index(name='数量')
        
        # 转换为字典形式
        return {row['价格区间']: row['数量'] for _, row in price_dist.iterrows()}
    except Exception as e:
        st.warning(f"创建价格分布统计时出错: {str(e)}")
        return {}

def create_change_distribution(df):
    """创建涨跌幅分布统计"""
    try:
        # 创建涨跌幅区间
        bins = 10
        change_range = np.linspace(df['涨跌幅'].min(), df['涨跌幅'].max(), bins + 1)
        labels = [f'{change_range[i]:.2f}%-{change_range[i+1]:.2f}%' for i in range(bins)]
        
        # 统计每个区间的股票数量
        df['涨跌幅区间'] = pd.cut(df['涨跌幅'], bins=bins, labels=labels)
        change_dist = df.groupby('涨跌幅区间').size().reset_index(name='数量')
        
        # 转换为字典形式
        return {row['涨跌幅区间']: row['数量'] for _, row in change_dist.iterrows()}
    except Exception as e:
        st.warning(f"创建涨跌幅分布统计时出错: {str(e)}")
        return {}

def main():
    st.set_page_config(page_title="A股数据分析系统", layout="wide", page_icon="")
    st.title("A股数据分析系统 （Tushare API 数据源）")
    
    # 顶部操作栏
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("基于 Tushare API 实时获取 A 股行情数据，支持多维度筛选和可视化分析")
    with col2:
        refresh = st.button("刷新数据", type="primary", use_container_width=True)
    
    # 数据加载与缓存
    if refresh or st.session_state['stock_data'].empty:
        with st.spinner("正在获取最新数据..."):
            df = fetch_stock_data()
            if not df.empty:
                st.session_state['stock_data'] = df
                st.session_state['last_updated'] = time.strftime("%Y-%m-%d %H:%M:%S")
                st.success(f"数据更新成功！最后更新时间：{st.session_state['last_updated']}")
            else:
                st.error("获取数据失败，请检查 Tushare Token 或网络连接")
    
    df = st.session_state['stock_data']
    if df.empty:
        st.warning("请点击刷新获取数据", icon="")
        return
    
    # 数据过滤面板
    st.subheader("数据过滤 ")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        name_filter = st.text_input("按股票名称/代码搜索", placeholder="输入关键词")
    with col2:
        price_min = st.number_input("最低价格", value=0.0, format="%.2f")
        price_max = st.number_input("最高价格", value=1000.0, format="%.2f")
    with col3:
        change_min = st.number_input("最小涨跌幅(%)", value=-100.0, format="%.1f")
        change_max = st.number_input("最大涨跌幅(%)", value=100.0, format="%.1f")
    with col4:
        volume_min = st.number_input("最小成交量(万手)", value=0.0, format="%.1f")
        pe_min = st.number_input("最小滚动市盈率", value=-100.0, format="%.1f")
    
    # 应用过滤条件
    filtered_df = df.copy()
    if name_filter:
        filtered_df = filtered_df[
            filtered_df['股票名称'].str.contains(name_filter, case=False, na=False) |
            filtered_df['股票代码'].str.contains(name_filter, na=False)
        ]
    
    filtered_df = filtered_df[
        (filtered_df['最新价'] >= price_min) &
        (filtered_df['最新价'] <= price_max) &
        (filtered_df['涨跌幅'] >= change_min) &
        (filtered_df['涨跌幅'] <= change_max) &
        (filtered_df['成交量'] >= volume_min * 10000) &  # 转换为手
        (filtered_df['滚动市盈率'] >= pe_min)
    ]
    
    # 数据展示
    st.subheader("实时行情数据 ")
    st.dataframe(
        filtered_df.style.format({
            '最新价': '¥{:.2f}',
            '涨跌幅': '{:.2f}%',
            '成交量': '{:,.0f}手',
            '成交额': '¥{:,.2f}万',
            '滚动市盈率': '{:.2f}',
            '市净率': '{:.2f}'
        }),
        use_container_width=True,
        height=400
    )
    
    # 数据导出
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="下载数据为CSV",
        data=csv,
        file_name=f"A股数据_{time.strftime('%Y%m%d_%H%M%S')}.csv",
        mime='text/csv',
        help="点击下载过滤后的数据"
    )
    
    # 市场概览
    st.subheader("市场概览 ")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("股票总数", len(df))
    with col2:
        up_count = len(df[df['涨跌幅'] > 0])
        st.metric("上涨家数", up_count)
    with col3:
        down_count = len(df[df['涨跌幅'] < 0])
        st.metric("下跌家数", down_count)
    with col4:
        flat_count = len(df[df['涨跌幅'] == 0])
        st.metric("平盘家数", flat_count)
    
    # 数据可视化
    st.subheader("数据可视化 ")
    col1, col2 = st.columns(2)
    
    with col1:
        # 价格分布统计
        price_dist = create_price_distribution(filtered_df)
        if price_dist:
            st.bar_chart(price_dist)
            st.caption("价格区间分布")
    
    with col2:
        # 涨跌幅分布统计
        change_dist = create_change_distribution(filtered_df)
        if change_dist:
            st.bar_chart(change_dist)
            st.caption("涨跌幅区间分布")
    
    # 热门股票
    st.subheader("热门股票 ")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 涨幅榜 TOP 10")
        st.dataframe(
            df.sort_values(by='涨跌幅', ascending=False).head(10)[['股票代码', '股票名称', '最新价', '涨跌幅']].style.format({
                '最新价': '¥{:.2f}',
                '涨跌幅': '{:.2f}%'
            }),
            use_container_width=True
        )
    
    with col2:
        st.markdown("### 跌幅榜 TOP 10")
        st.dataframe(
            df.sort_values(by='涨跌幅', ascending=True).head(10)[['股票代码', '股票名称', '最新价', '涨跌幅']].style.format({
                '最新价': '¥{:.2f}',
                '涨跌幅': '{:.2f}%'
            }),
            use_container_width=True
        )

if __name__ == "__main__":
    main()