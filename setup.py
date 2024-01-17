from setuptools import setup, find_packages

setup(
    name='equilibrium_scipy',
    version="1.0.1",  # バージョン
    description="calculating equilibrium points & stability analysis by scipy",  # 説明
    author='poteto0',  # 作者名
    packages=find_packages(),  # 使うモジュール一覧を指定する
    license='MIT'  # ライセンス
)