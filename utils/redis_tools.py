# -*- encoding: utf-8 -*-

from common.consts import java_conf
import redis

pool = redis.ConnectionPool(**java_conf['udev']['redis'])
# redis.Redis
r_cli = redis.StrictRedis(connection_pool=pool, decode_responses=True)


# r_cli.pipeline()

def exs_k(*names):
    return r_cli.exists(*names)

def del_k(*names):
    return r_cli.delete(*names)

def h_has_key(name, key):
    return r_cli.hexists(name, key)


def h_set(name, key=None, value=None, java=False, mapping=None):
    if java:
        r_cli.hset(name, key, '"' + value + '"', mapping)
    else:
        r_cli.hset(name, key, value, mapping)
    return True


def h_get(name, key, java=False):
    res = r_cli.hget(name, key)
    if res:
        res = str(res, encoding='utf-8')
        if res and res.startswith('"') and res.endswith('"'):
            res = res.strip('"')
    return res


def h_getall(name):
    res = r_cli.hgetall(name)
    data = {}
    if res:

        # res = {key.decode('utf-8'): value for key, value in res.items()}
        for k, v in res.items():
            v = str(v, encoding='utf-8')
            # if v
            if v and v.startswith('"') and v.endswith('"'):
                data[k.decode('utf-8')] = v.strip('"')
            else:
                data[k.decode('utf-8')] = v
    return data


def h_get_to_int(name, key):
    return int(r_cli.hget(name, key))


def h_del(name, key):
    r_cli.hdel(name, key)
    return True


def s_add(name, *values, java=False):
    # set
    if java:
        new_values = ['"' + v + '"' if isinstance(v, str) else v for v in values]
    else:
        new_values = values
    r_cli.sadd(name, *new_values)


def s_has_key(name, value, java=False):
    if java:
        return r_cli.sismember(name, '"' + value + '"' if isinstance(value, str) else value)
    else:
        return r_cli.sismember(name, value)


def set_remove(java, name, *values):
    """
    如果是java的RedisTemplate存的字符串，由于序列化问题，
    会保存成 ”\"value\"“ 形式，所以需要加 双上引号.
    java 没给默认值是怕埋坑，所以还是每次必须传吧
    如果哪天 java 那边的value的序列化方式改了，这里也得改,或者分开

    Args:
        java: 是否采用java的序列化方式
        name:
        values: 可以是单个值，也可以是可迭代对象
        
    """
    if java:
        new_values = ['"' + v + '"' if isinstance(v, str) else v for v in values]
    else:
        new_values = values

    r_cli.srem(name, *new_values)


def s_set(name, value, java=False, **kwargs):
    # string
    if java:
        r_cli.set(name, '"' + value + '"' if isinstance(value, str) else value, **kwargs)
    else:
        r_cli.set(name, value, **kwargs)


def s_get(name):
    res = r_cli.get(name)
    if isinstance(res, str) and res.startswith('"') and res.endswith('"'):
        res = res.strip('"')
    return res


def s_remove(name):
    r_cli.delete(name)


def s_exists(*names):
    return r_cli.exists(*names)

# def r_push(name, *values, java=False):
#     return  r_cli.rpush(name, *values)

if __name__ == '__main__':
    print(exs_k("suc_2167768208b911ed8f7c024202cdb4fb"))
    # h_set('papa','pid', 235)
    # h_set('papa','type', 'rec')
    # m = h_getall('papa')
    # print(m)
    # del_k('papa')
    # m = h_getall('papa')
    # if m:
    #     print(1)
    # else:
    #     print(2)
    # r_push('papa',34,'edf')
    # r_cli.sadd()
    # r_cli.set()
    # r_cli.sadd("abc", "345")
    # s_add("abc", "dd")
    # set_remove(False, "abc", "dd")
    # s_add("abc", "rr")
    # print(r_cli.sismember("abc", "rr"))
    # r_cli.srem(False, "abc", "rr")
    # m = ["abc","sdfa","sdfd"]
    # set_remove(True, '123', "abc")
    # set_remove(True, '123', *m)
    # s_set('completed_epoch', 7)
    # s_set('completed_epoch2', 7)
    # s_set('completed_epoch', 8)
    # print(s_exists('completed_epoch', 'completed_epoch2'))
    # s_remove('completed_epoch')
    # print(int(s_get('abcdedf')))
    # s_set("ttt", json.dumps({
    #     'a': 1,
    #     'b': 2,
    #     'c': {
    #         'd': 4
    #     }
    # }))
    # p = s_get("ttt")
    # q = json.loads(p)
    # print(q['c']['d'])

    # m = ['a','b','c']
    # h_set('test','array','a,b,c')
    # v = h_get('test','array')
    # v = v.split(',')
    # print(v)
    # m = s_get("pppp")
    # if m:
    #     print('getit')
    # else:
    #     print('notgetit')