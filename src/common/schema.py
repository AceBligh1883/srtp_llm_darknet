# src/common/schema.py

# 定义我们希望从文本中提取的实体类型及其描述
# 这将指导LLM进行命名实体识别
ENTITY_TYPES = {
    # 攻击者与基础设施
    "ORGANIZATION": "组织、团伙、APT组织或勒索软件团伙 (e.g., Conti, APT29)",
    "PERSON": "个人，如黑客、开发者、管理员、中间人 (e.g., John Doe, ShinyHunters)",
    "MALWARE": "恶意软件家族，如勒索软件、间谍软件、加载器 (e.g., DarkSide, Emotet)",
    "HACKING_TOOL": "黑客工具，如漏洞利用套件、扫描器、C2框架 (e.g., Metasploit, Cobalt Strike)",
    "TTP": "战术、技术和过程，通常是ATT&CK框架的ID (e.g., T1566, T1059.001)",
    "DOMAIN_NAME": "域名 (e.g., evil-c2-server.com)",
    "IP_ADDRESS": "IPv4或IPv6地址 (e.g., 192.168.1.100)",
    "URL": "完整的统一资源定位符 (e.g., http://phishing.site/login.php)",
    
    # 攻击目标与受害者
    "VICTIM_ORG": "受害者组织名称 (e.g., ACME Corp)",
    "VICTIM_INDUSTRY": "受害者所属行业 (e.g., Healthcare, Financial Services)",
    "SOFTWARE": "被攻击或被利用的软件、产品或系统 (e.g., Microsoft Exchange, Apache Log4j)",
    "VULNERABILITY": "软件漏洞编号 (e.g., CVE-2021-44228, Log4Shell)",

    # 非法商品与服务
    "WEAPON": "武器名称或型号 (e.g., AK-47, Glock-19)",
    "DRUG": "毒品或化学品名称 (e.g., Fentanyl, Methamphetamine)",
    "STOLEN_DATA": "被盗数据或数据泄露的类型 (e.g., Credit Card Data, PII, Login Credentials)",
    "ILLICIT_SERVICE": "提供的非法服务 (e.g., DDoS-for-hire, Money Laundering)",
    
    # 金融与交易
    "CRYPTOCURRENCY": "加密货币名称 (e.g., Bitcoin, Monero)",
    "WALLET_ADDRESS": "加密货币钱包地址",
    "TRANSACTION_ID": "加密货币交易哈希或ID",
    "DARKNET_MARKET": "暗网市场名称 (e.g., AlphaBay, Silk Road)",
    "PRICE": "价格或金额 (e.g., $500, 0.5 BTC)",

    # 身份与凭证
    "EMAIL": "电子邮件地址",
    "USERNAME": "用户名或昵称",
    "PGP_KEY": "PGP公钥或指纹",
    
    # 通用
    "LOCATION": "地理位置，如城市或国家 (e.g., Russia, USA)",
}

# 定义我们希望提取的关系类型
# 这将指导LLM进行关系抽取
RELATION_TYPES = [
    # 攻击与利用
    "TARGETS",          # (Attacker -> Victim): 攻击目标是
    "EXPLOITS",         # (Attacker/Malware -> Vulnerability): 利用漏洞
    "COMPROMISES",      # (Attacker -> Victim): 成功入侵
    "DELIVERS",         # (Malware_A -> Malware_B): 投递/分发
    "DOWNLOADS_FROM",   # (Victim/Malware -> URL/IP): 从...下载
    
    # 归属与关联
    "ATTRIBUTED_TO",    # (Organization/Person -> Location/Country): 归因于
    "OPERATES",         # (Person/Organization -> Malware/Market): 运营/操作
    "PART_OF",          # (Person -> Organization): 是...的一部分
    "COMMUNICATES_WITH",# (Person/Organization -> Person/Organization): 与...通信
    "DEVELOPED",        # (Person/Organization -> Malware/Tool): 开发了
    
    # 基础设施与工具
    "USES",             # (Attacker -> Tool/Malware/TTP): 使用
    "HOSTS",            # (IP/Domain -> Domain/URL): 托管
    "HAS_C2",           # (Malware -> Domain/IP): 命令与控制服务器是
    
    # 交易与金融
    "SELLS",            # (Seller -> Item): 销售
    "BUYS",             # (Buyer -> Item): 购买
    "HAS_PRICE",        # (Item -> Price): 价格是
    "HAS_WALLET",       # (Person/Organization -> WalletAddress): 拥有钱包
    "SENDS_FUNDS_TO",   # (Wallet_A -> Wallet_B): 资金流向
    
    # 分类与属性
    "IS_A",             # (Instance -> Class): 是一个/是一种
    "HAS_EMAIL",        # (Person/Organization -> Email): 拥有邮箱
    "HAS_VULNERABILITY",# (Software -> Vulnerability): 存在漏洞
    "LOCATED_IN",       # (Entity -> Location): 位于
    "ASSOCIATED_WITH",  # 通用关联，作为备用
]