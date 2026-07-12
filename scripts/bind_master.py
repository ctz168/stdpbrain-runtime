#!/usr/bin/env python3
"""
bind_master.py — 让 brain 智能体加主人 (1000008) 为好友
=================================================

流程:
1. 加载 brain agent 的 AICQ 身份 (从 ~/.aicq-sdk/loop/identity.json)
   - 如果不存在, 创建新身份
2. 注册 + 登录, 拿到 access_token
3. 查询好友列表, 看主人 (1000008) 是否已是好友
4. 如果不是, 发送好友请求 (附验证消息)
5. 生成二维码 (主人扫码可反向加 brain), 方便绑定
6. 轮询好友请求状态, 直到主人接受
7. 接受后, brain 就能给主人发消息了

用法:
    python3 bind_master.py --master-id 1000008
    python3 bind_master.py --master-id 1000008 --qr qr.png  # 生成二维码
"""

from __future__ import annotations
import os, sys, json, asyncio, argparse, time, qrcode
from pathlib import Path

# 路径
AICQSDK_DIR = "/home/z/my-project/repos/AIcqsdk"
RUNTIME_DIR = "/home/z/my-project/brain_runtime"
sys.path.insert(0, AICQSDK_DIR)
os.environ["HOME"] = RUNTIME_DIR  # 让 aicq SDK 把身份文件放在 runtime

import aiohttp
from aicq.loop import (
    _get_or_create_identity, _ensure_registered, _login,
    _http_get, _http_post, IDENTITY_FILE
)
from aicq import crypto

AICQ_SERVER = "https://aicq.me"


async def bind_master(master_id: str, qr_path: str = "", message: str = ""):
    """让 brain agent 加 master_id 为好友."""

    print("=" * 70)
    print(f"🤝 brain agent 绑定主人")
    print(f"   master_id: {master_id}")
    print(f"   server: {AICQ_SERVER}")
    print(f"   identity file: {IDENTITY_FILE}")
    print("=" * 70)

    # 1. 加载/创建身份
    print("\n[1/6] 加载 brain agent 身份...")
    identity = _get_or_create_identity()
    print(f"  ✅ account_id: {identity.get('account_id', '(未注册)')}")
    print(f"     signing_pub: {identity['signing_pub'][:32]}...")

    async with aiohttp.ClientSession() as session:
        # 2. 注册 + 登录
        print("\n[2/6] 注册 + 登录...")
        await _ensure_registered(session, AICQ_SERVER, identity)
        print(f"  ✅ registered: {identity.get('account_id')}")

        access_token = await _login(session, AICQ_SERVER, identity)
        print(f"  ✅ logged in, access_token: {access_token[:20]}...")

        headers = {"Authorization": f"Bearer {access_token}"}

        # 3. 查询好友列表
        print(f"\n[3/6] 查询好友列表, 看主人 {master_id} 是否已是好友...")
        try:
            result = await _http_get(session, f"{AICQ_SERVER}/api/v1/friends", headers=headers)
            friends = result.get("friends") or result.get("data", [])
            print(f"  当前好友数: {len(friends)}")
            for f in friends[:5]:
                print(f"    - {f.get('account_id', '?')} ({f.get('name', f.get('display_name', '?'))})")

            # 检查主人是否在好友列表
            already_friend = any(
                str(f.get("account_id", "")) == str(master_id) or
                str(f.get("id", "")) == str(master_id) or
                str(f.get("friend_id", "")) == str(master_id)
                for f in friends
            )
            if already_friend:
                print(f"\n  ✅ 主人 {master_id} 已是好友! 可以直接发消息.")
                # 生成二维码备用
                if qr_path:
                    _generate_qr(identity, qr_path)
                return True
            print(f"  主人 {master_id} 不在好友列表, 需要发送好友请求.")
        except Exception as e:
            print(f"  ⚠️ 查询好友列表失败: {e}")

        # 4. 发送好友请求
        print(f"\n[4/6] 发送好友请求给 {master_id}...")
        msg = message or f"我是你的 brain agent ({identity.get('account_id', '?')}), 请加我为好友, 我想给你发学习汇报!"
        try:
            result = await _http_post(
                session, f"{AICQ_SERVER}/api/v1/friends/request",
                {"to_id": master_id, "message": msg}, headers=headers
            )
            print(f"  ✅ 好友请求已发送: {result}")
            request_id = result.get("request_id") or result.get("id", "")
            if request_id:
                print(f"     request_id: {request_id}")
        except Exception as e:
            err_str = str(e)
            if "already" in err_str.lower() or "exists" in err_str.lower() or "pending" in err_str.lower():
                print(f"  ℹ️ 好友请求已存在 (pending), 等待主人接受")
            else:
                print(f"  ❌ 发送好友请求失败: {e}")
                return False

        # 5. 生成二维码 (主人扫码反向加 brain)
        if qr_path:
            print(f"\n[5/6] 生成二维码, 主人扫码可反向加 brain 为好友...")
            _generate_qr(identity, qr_path)
        else:
            print(f"\n[5/6] 跳过二维码生成")

        # 6. 轮询等待主人接受 (最多 5 分钟)
        print(f"\n[6/6] 等待主人 {master_id} 接受好友请求 (最多 5 分钟)...")
        print(f"   请主人在 AICQ 客户端查看好友请求并接受.")
        print(f"   或扫描二维码: {qr_path if qr_path else '(未生成)'}")

        deadline = time.time() + 300  # 5 min
        last_check = 0
        while time.time() < deadline:
            # 每 10 秒查一次
            if time.time() - last_check < 10:
                await asyncio.sleep(2)
                continue
            last_check = time.time()

            try:
                # 查好友列表
                result = await _http_get(session, f"{AICQ_SERVER}/api/v1/friends", headers=headers)
                friends = result.get("friends") or result.get("data", [])
                is_friend_now = any(
                    str(f.get("account_id", "")) == str(master_id) or
                    str(f.get("id", "")) == str(master_id)
                    for f in friends
                )
                if is_friend_now:
                    print(f"\n  🎉 主人 {master_id} 已接受好友请求!")
                    print(f"  brain agent 现在可以给主人发消息了.")
                    return True
                # 查好友请求状态
                try:
                    reqs = await _http_get(session, f"{AICQ_SERVER}/api/v1/friends/requests", headers=headers)
                    sent = reqs.get("sent", [])
                    pending = [r for r in sent if str(r.get("to_id", "")) == str(master_id) and r.get("status") == "pending"]
                    if pending:
                        elapsed = int(time.time() - (deadline - 300))
                        print(f"  [{elapsed}s] 仍在等待主人接受... (pending)")
                    else:
                        # 不在 pending 了, 可能被接受或拒绝
                        accepted = [r for r in sent if str(r.get("to_id", "")) == str(master_id) and r.get("status") == "accepted"]
                        if accepted:
                            print(f"\n  🎉 主人 {master_id} 已接受好友请求!")
                            return True
                        # 重新查好友列表确认
                        result = await _http_get(session, f"{AICQ_SERVER}/api/v1/friends", headers=headers)
                        friends = result.get("friends") or result.get("data", [])
                        if any(str(f.get("account_id", "")) == str(master_id) or str(f.get("id", "")) == str(master_id) for f in friends):
                            print(f"\n  🎉 主人 {master_id} 已是好友!")
                            return True
                except Exception as e:
                    print(f"  (查询请求状态失败: {e}, 继续等待)")
            except Exception as e:
                print(f"  (查询失败: {e}, 重试)")

        print(f"\n  ⏰ 5 分钟超时, 主人未接受. brain 仍可发送请求, 等主人后续接受.")
        return False


def _generate_qr(identity: dict, qr_path: str):
    """生成二维码: aicq-master-v1:{signing_sec}:{account_id}:{signing_pub}"""
    signing_sec = identity.get("signing_sec", "")
    account_id = identity.get("account_id", "")
    signing_pub = identity.get("signing_pub", "")
    payload = f"aicq-master-v1:{signing_sec}:{account_id}:{signing_pub}"

    img = qrcode.make(payload)
    img.save(qr_path)
    print(f"  ✅ 二维码已保存: {qr_path}")
    print(f"     account_id: {account_id}")
    print(f"     主人用 AICQ app 扫描此二维码, 即可绑定 brain 为好友")


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--master-id", default="1000008", help="主人 AICQ account id")
    parser.add_argument("--qr", default="", help="二维码图片保存路径")
    parser.add_argument("--message", default="", help="好友请求验证消息")
    args = parser.parse_args()

    qr_path = args.qr or os.path.join(RUNTIME_DIR, "brain_agent_qr.png")
    success = await bind_master(args.master_id, qr_path, args.message)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
