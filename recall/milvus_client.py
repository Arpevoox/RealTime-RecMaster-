"""
Milvus 客户端封装
用于 RealTime-RecMaster 项目的向量存储和检索
"""

from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType


class MilvusClientWrapper:
    """
    Milvus 客户端包装类，用于存储和检索物品嵌入
    使用 HNSW 索引实现高效的相似性搜索
    """
    def __init__(self, uri="http://localhost:19530", collection_name="item_embeddings"):
        self.client = MilvusClient(uri=uri)
        self.collection_name = collection_name
        self.dim = 128
        
        # 创建集合（如果不存在）
        self._create_collection_if_not_exists()
    
    def _create_collection_if_not_exists(self):
        """
        创建 Milvus 集合（如果不存在）
        使用 HNSW 索引提高搜索效率
        """
        collections = self.client.list_collections()
        
        if self.collection_name not in collections:
            # 定义字段
            fields = [
                FieldSchema(name="item_id", dtype=DataType.INT64, is_primary=True),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.dim)
            ]
            
            # 创建集合模式
            schema = CollectionSchema(fields, description="Item embeddings collection")
            
            # 创建集合
            self.client.create_collection(collection_name=self.collection_name, schema=schema)
            
            # 创建 HNSW 索引 (分层小世界图索引，高效近似最近邻搜索)
            index_params = {
                "index_type": "HNSW",  # 使用 HNSW 索引
                "metric_type": "COSINE",  # 使用余弦相似度
                "params": {
                    "M": 16,           # HNSW 图的 M 参数，控制每个层的连接数
                    "efConstruction": 200  # 建立索引时的 ef 参数，影响索引质量
                }
            }
            self.client.create_index(collection_name=self.collection_name, index_params=index_params)
    
    def upsert_item_embeddings(self, item_ids, embeddings):
        """
        插入或更新物品嵌入向量
        """
        data = [
            {
                "item_id": int(item_id),
                "embedding": embedding.tolist() if isinstance(embedding, torch.Tensor) else embedding
            }
            for item_id, embedding in zip(item_ids, embeddings)
        ]
        
        # 删除现有数据（如果需要）
        for record in data:
            self.client.delete(collection_name=self.collection_name, filter=f"item_id == {record['item_id']}")
        
        # 插入新数据
        self.client.insert(collection_name=self.collection_name, data=data)
        print(f"Upserted {len(data)} item embeddings to Milvus")
    
    def recall_for_user(self, user_embedding, top_k=500):
        """
        根据用户嵌入向量召回最相似的物品
        使用 HNSW 索引进行高效搜索
        """
        # 搜索参数 (针对 HNSW 索引优化)
        search_params = {
            "metric_type": "COSINE",
            "params": {
                "ef": 64  # 搜索时的 ef 参数，影响搜索精度和速度的平衡
            }
        }
        
        # 执行搜索
        results = self.client.search(
            collection_name=self.collection_name,
            data=[user_embedding.tolist() if isinstance(user_embedding, torch.Tensor) else user_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            output_fields=["item_id"]
        )
        
        # 提取结果
        recalled_items = []
        scores = []
        if results and len(results) > 0:
            for hit in results[0]:
                recalled_items.append(hit['entity']['item_id'])
                scores.append(hit['distance'])
        
        return recalled_items[:top_k], scores[:top_k]