# τ-bench 数据库设计文档

## 概述

τ-bench 采用了基于 JSON 文件的轻量级数据存储方案，而非传统的关系型数据库。这种设计专门为 LLM 工具交互基准测试优化，提供了简单、高效且可重现的数据管理解决方案。

## 核心设计思路

### 1. 无传统数据库的设计理念

τ-bench **没有使用传统的数据库系统**（如MySQL、PostgreSQL等），而是采用了：

- **JSON 文件存储**：所有数据以 JSON 格式存储在文件系统中
- **内存数据操作**：运行时将数据加载到内存中进行操作
- **无状态设计**：每次测试都从干净的数据状态开始

### 2. 设计优势

这种设计选择的核心优势包括：

- **简单性**：无需配置复杂的数据库环境
- **可重现性**：每次测试都使用相同的初始数据状态
- **便携性**：整个数据集可以轻松打包和分发
- **透明性**：数据格式直观，便于调试和验证
- **快速重置**：可以快速回滚到初始状态

## 数据架构设计

### 整体架构

```
τ-bench/
├── tau_bench/envs/
│   ├── retail/data/           # 零售环境数据
│   │   ├── users.json         # 用户数据
│   │   ├── products.json      # 产品数据
│   │   └── orders.json        # 订单数据
│   └── airline/data/          # 航空环境数据
│       ├── users.json         # 用户数据
│       ├── flights.json       # 航班数据
│       └── reservations.json  # 预订数据
```

### 数据加载机制

每个环境都有统一的数据加载函数：

```python
def load_data() -> dict[str, Any]:
    # 从JSON文件加载所有数据到内存
    return {
        "users": user_data,
        "products": product_data,
        "orders": order_data,
    }
```

## 领域数据模型

### 零售环境 (Retail Domain)

#### 用户模型 (Users)
```json
{
    "user_id": {
        "name": {"first_name": "string", "last_name": "string"},
        "address": {
            "address1": "string",
            "address2": "string", 
            "city": "string",
            "country": "string",
            "state": "string",
            "zip": "string"
        },
        "email": "string",
        "payment_methods": {
            "method_id": {
                "source": "paypal|credit_card|gift_card",
                "brand": "string", // for credit cards
                "last_four": "string", // for credit cards
                "id": "string"
            }
        },
        "orders": ["order_id_list"]
    }
}
```

#### 产品模型 (Products)
```json
{
    "product_id": {
        "name": "string",
        "product_id": "string",
        "variants": {
            "item_id": {
                "item_id": "string",
                "options": {
                    "color": "string",
                    "size": "string",
                    "material": "string",
                    "style": "string"
                },
                "available": boolean,
                "price": number
            }
        }
    }
}
```

#### 订单模型 (Orders)
```json
{
    "order_id": {
        "order_id": "string",
        "user_id": "string",
        "address": {}, // 配送地址
        "items": [
            {
                "name": "string",
                "product_id": "string",
                "item_id": "string",
                "price": number,
                "options": {}
            }
        ],
        "status": "pending|shipped|delivered|cancelled",
        "total": number,
        "payment_method": {},
        "shipping_date": "string",
        "delivery_date": "string"
    }
}
```

### 航空环境 (Airline Domain)

#### 用户模型 (Users)
```json
{
    "user_id": {
        "name": {"first_name": "string", "last_name": "string"},
        "email": "string",
        "phone": "string",
        "address": {},
        "reservations": ["reservation_id_list"]
    }
}
```

#### 航班模型 (Flights)
```json
{
    "flight_number": {
        "flight_number": "string",
        "origin": "string", // 机场代码
        "destination": "string",
        "scheduled_departure_time_est": "string",
        "scheduled_arrival_time_est": "string",
        "dates": {
            "YYYY-MM-DD": {
                "status": "scheduled|landed|cancelled|delayed",
                "actual_departure_time_est": "string",
                "actual_arrival_time_est": "string"
            }
        }
    }
}
```

#### 预订模型 (Reservations)
```json
{
    "reservation_id": {
        "reservation_id": "string",
        "user_id": "string",
        "flights": [
            {
                "flight_number": "string",
                "date": "string",
                "seat": "string"
            }
        ],
        "passengers": [
            {
                "name": {"first_name": "string", "last_name": "string"},
                "age": number,
                "passport": "string"
            }
        ],
        "status": "confirmed|cancelled",
        "total_cost": number,
        "baggage": {}
    }
}
```

## 数据操作层

### 工具系统 (Tool System)

数据操作通过工具（Tool）系统实现：

```python
class Tool(abc.ABC):
    @staticmethod
    def invoke(data: Dict[str, Any], **kwargs) -> str:
        # 直接操作内存中的数据字典
        pass
    
    @staticmethod
    def get_info() -> Dict[str, Any]:
        # 返回工具的API描述
        pass
```

### 数据一致性保证

- **哈希验证**：使用 SHA256 哈希确保数据状态的一致性
- **原子操作**：每个工具调用都是原子性的
- **状态重置**：测试结束后自动重置到初始状态

```python
def get_data_hash(self) -> str:
    return consistent_hash(to_hashable(self.data))

def calculate_reward(self) -> RewardResult:
    # 验证数据变更是否符合预期
    data_hash = self.get_data_hash()
    # 执行ground truth动作
    # 比较最终状态哈希
    gt_data_hash = self.get_data_hash()
    reward = 1.0 if data_hash == gt_data_hash else 0.0
```

## Mock 数据生成策略

### 生成流程

1. **模式设计**：首先由人工设计数据库模式和字段结构
2. **种子数据生成**：使用 GPT 生成多样化的种子数据（姓名、地址、产品类型等）
3. **程序化组合**：通过代码将种子数据与算法生成的数据组合
4. **关系构建**：建立用户-订单、产品-变体等关系

### 数据特点

- **真实性**：模拟真实世界的业务场景
- **多样性**：确保数据的多样性和代表性
- **一致性**：保持数据间的逻辑一致性
- **可扩展性**：易于添加新的数据类型和字段

## Action执行时的数据操作机制

### 核心原理

**重要发现**：τ-bench在执行action时，**不是直接读写JSON文件**，而是采用以下机制：

1. **内存数据操作**：所有数据操作都在内存中的Python字典上进行
2. **无文件I/O**：工具调用过程中不会产生任何文件读写操作
3. **会话级状态**：数据状态在整个会话期间保持在内存中

### 数据状态的定义和生命周期

#### 初始状态的定义

**初始状态**是指每个测试任务开始时的标准数据状态，它具有以下特征：

1. **静态性**：初始状态永远不变，总是从同一组JSON文件加载
2. **一致性**：无论执行多少次测试，初始状态都完全相同
3. **完整性**：包含所有业务实体的完整数据集
4. **清洁性**：不包含任何测试过程中产生的临时数据或状态

#### 初始状态示例

以零售环境为例，初始状态包含：

```python
# 初始状态数据结构示例
initial_state = {
    "users": {
        "noah_brown_6181": {
            "name": {"first_name": "Noah", "last_name": "Brown"},
            "address": {
                "address1": "986 Sunset Drive",
                "city": "Denver",
                "state": "CO",
                "zip": "80279"
            },
            "email": "noah.brown7922@example.com",
            "payment_methods": {
                "paypal_5727330": {"source": "paypal", "id": "paypal_5727330"},
                "credit_card_7815826": {
                    "source": "credit_card",
                    "brand": "mastercard", 
                    "last_four": "9212"
                }
            },
            "orders": ["#W7678072"]  # 已存在的历史订单
        }
    },
    "orders": {
        "#W7678072": {
            "order_id": "#W7678072",
            "user_id": "noah_brown_6181",
            "status": "delivered",  # 初始状态为已完成的订单
            "items": [/*商品列表*/],
            "total": 245.67,
            "delivery_date": "2024-04-20"
        }
    },
    "products": {
        "9523456873": {
            "name": "T-Shirt",
            "variants": {
                "9612497925": {
                    "options": {"color": "blue", "size": "M"},
                    "available": true,  # 初始库存状态
                    "price": 50.88
                }
            }
        }
    }
}
```

#### 状态初始化
```python
class Env:
    def __init__(self, data_load_func: Callable[[], Dict[str, Any]], ...):
        self.data_load_func = data_load_func
        self.data = data_load_func()  # 从JSON文件一次性加载到内存
```

#### 初始状态 vs 运行时状态对比

```python
# 初始状态：任务开始时
initial_state = {
    "users": {
        "noah_brown_6181": {
            "payment_methods": {
                "gift_card_4374071": {
                    "source": "gift_card",
                    "balance": 500.00  # 初始余额
                }
            },
            "orders": ["#W7678072"]  # 只有历史订单
        }
    },
    "orders": {
        "#W7678072": {
            "status": "delivered",  # 已完成的历史订单
            "total": 245.67
        }
    }
}

# 运行时状态：执行action后
runtime_state = {
    "users": {
        "noah_brown_6181": {
            "payment_methods": {
                "gift_card_4374071": {
                    "source": "gift_card", 
                    "balance": 350.00  # 余额减少（用于新订单）
                }
            },
            "orders": ["#W7678072", "#W8892341"]  # 添加了新订单
        }
    },
    "orders": {
        "#W7678072": {
            "status": "delivered",  # 历史订单保持不变
            "total": 245.67
        },
        "#W8892341": {  # 新创建的订单
            "order_id": "#W8892341",
            "user_id": "noah_brown_6181", 
            "status": "pending",
            "total": 150.00,
            "items": [{"name": "New T-Shirt", "price": 150.00}]
        }
    }
}
```

#### 状态重置机制
```python
def reset(self, task_index: Optional[int] = None):
    self.data = self.data_load_func()  # 重新从文件加载，覆盖内存状态
    # 此时 runtime_state 被丢弃，系统回到 initial_state
```

#### Action执行时的数据操作
```python
def step(self, action: Action) -> EnvResponse:
    if action.name in self.tools_map:
        # 工具直接操作内存中的数据字典
        observation = self.tools_map[action.name].invoke(
            data=self.data, **action.kwargs  # self.data是内存中的字典
        )
```

### 具体的数据读写实现

#### 数据读取工具示例
```python
class GetOrderDetails(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str) -> str:
        orders = data["orders"]  # 直接访问内存字典
        if order_id in orders:
            return json.dumps(orders[order_id])  # 序列化返回
        return "Error: order not found"
```

#### 数据修改工具示例
```python
class ModifyPendingOrderItems(Tool):
    @staticmethod
    def invoke(data: Dict[str, Any], order_id: str, ...):
        orders = data["orders"]  # 获取内存中的订单字典
        
        # 直接修改内存中的数据结构
        order = orders[order_id]
        order["status"] = "pending (item modified)"
        
        # 修改嵌套的商品信息
        for item_id, new_item_id in zip(item_ids, new_item_ids):
            item = [item for item in order["items"] if item["item_id"] == item_id][0]
            item["item_id"] = new_item_id
            item["price"] = products[item["product_id"]]["variants"][new_item_id]["price"]
        
        return json.dumps(order)  # 返回修改后的数据
```

### 数据一致性和状态管理

#### 状态哈希验证
```python
def get_data_hash(self) -> str:
    return consistent_hash(to_hashable(self.data))  # 对整个内存状态计算哈希

def calculate_reward(self) -> RewardResult:
    data_hash = self.get_data_hash()  # 记录当前状态
    # 执行ground truth操作
    self.data = self.data_load_func()  # 重置到初始状态
    for action in self.task.actions:
        self.step(action)  # 重新执行期望的操作序列
    gt_data_hash = self.get_data_hash()  # 计算期望的最终状态
    
    # 比较实际状态与期望状态
    reward = 1.0 if data_hash == gt_data_hash else 0.0
```

#### 原子性保证
- 每个工具调用都是原子性的
- 要么完全成功并修改数据，要么完全失败不产生任何变更
- 通过异常处理确保数据一致性

#### 并发安全
- 每个测试实例都有独立的数据副本
- 不同任务间通过数据重置确保隔离
- 支持并发测试而不会产生数据竞争

### 内存数据结构

执行期间的数据在内存中表现为嵌套的Python字典：

```python
self.data = {
    "users": {
        "user_id_1": { ... },
        "user_id_2": { ... }
    },
    "orders": {
        "order_id_1": { ... },
        "order_id_2": { ... }
    },
    "products": { ... }
}
```

所有工具都通过引用直接操作这个内存结构，实现了：
- **零延迟**的数据访问
- **实时一致性**的状态更新
- **完整的事务性**操作

## 技术实现细节

### 数据加载与缓存

```python
class Env:
    def __init__(self, data_load_func: Callable[[], Dict[str, Any]], ...):
        self.data_load_func = data_load_func
        self.data = data_load_func()  # 初始加载
    
    def reset(self, task_index: Optional[int] = None):
        self.data = self.data_load_func()  # 重新加载干净数据
```

### 环境隔离

- 每个环境（airline、retail）都有独立的数据空间
- 不同任务间通过数据重置确保隔离
- 支持并发测试而不会产生数据竞争

### 性能优化

- **内存操作**：所有数据操作都在内存中进行，避免I/O开销
- **懒加载**：按需加载数据文件
- **批量处理**：支持批量任务处理

## 扩展性设计

### 新环境添加

添加新的业务环境需要：

1. 创建 `envs/new_domain/` 目录
2. 设计数据模型并创建 JSON 文件
3. 实现 `load_data()` 函数
4. 开发相应的工具集合

### 数据模型扩展

- JSON 格式天然支持模式演化
- 可以轻松添加新字段而不影响现有工具
- 支持嵌套结构和复杂数据类型

## 总结

τ-bench 的数据库设计是一个专为 LLM 基准测试优化的轻量级解决方案。通过放弃传统数据库，采用 JSON 文件 + 内存操作的模式，实现了：

- **零配置**：无需数据库安装和配置
- **完全可重现**：每次测试都从相同状态开始
- **高性能**：内存操作确保低延迟
- **易调试**：数据格式直观透明
- **强扩展性**：支持快速添加新的业务领域

这种设计特别适合需要频繁重置、高度可控的基准测试场景，为 LLM 工具使用能力的评估提供了理想的数据基础设施。
