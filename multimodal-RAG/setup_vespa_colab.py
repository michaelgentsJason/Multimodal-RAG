#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
在Google Colab中设置和部署Vespa
"""

import os
import time
import subprocess
import requests
import json
from pyvespa import Vespa, Document, Field, ApplicationPackage
from pyvespa import FieldType, RankProfile, Schema


def setup_vespa():
    """安装和部署Vespa环境"""
    print("开始设置Vespa环境...")

    # 1. 安装pyvespa和其他依赖
    subprocess.run(["pip", "install", "pyvespa"], check=True)

    # 2. 使用pyvespa创建应用
    app_package = ApplicationPackage(name="multimodal")

    # 3. 定义文档模式
    schema = Schema(
        name="multimodal_document",
        fields=[
            Field(name="text_content", type=FieldType.STRING, indexing=["SUMMARY", "INDEX"], matching=["TEXT"]),
            Field(name="text_embedding", type=FieldType.TENSOR(dimensions=["x[768]"]), indexing=["ATTRIBUTE", "INDEX"],
                  attribute={"distance-metric": "angular"}),
            Field(name="image_embedding", type=FieldType.TENSOR(dimensions=["x[1024]"]),
                  indexing=["ATTRIBUTE", "INDEX"],
                  attribute={"distance-metric": "angular"}),
            Field(name="page_index", type=FieldType.INT, indexing=["SUMMARY", "ATTRIBUTE"]),
            Field(name="pdf_path", type=FieldType.STRING, indexing=["SUMMARY", "ATTRIBUTE"]),
        ]
    )

    # 4. 定义排序配置
    rank_profile = RankProfile(
        name="default",
        inputs=[("text_weight", "double"), ("image_weight", "double")],
        first_phase="query(text_weight) * closeness(field, text_embedding) + query(image_weight) * closeness(field, image_embedding)"
    )

    # 5. 添加HNSW配置
    schema.add_hnsw_index("text_embedding")
    schema.add_hnsw_index("image_embedding")

    # 6. 添加到应用包
    app_package.add_schema(schema)

    # 7. 创建并部署应用
    vespa_app = Vespa(app_package, deploy=True)

    print("Vespa环境设置完成！")
    return vespa_app


if __name__ == "__main__":
    vespa_app = setup_vespa()
    print("Vespa应用已成功部署，可以开始使用了！")