"""
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Ontology Engineering Group
        http://www.oeg-upm.net/
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Copyright (C) 2017 Ontology Engineering Group.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
#-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=#
"""

import re

import jsonpath_rw
import operator
from jsonpath_ng import DatumInContext
from jsonpath_ng.ext import parser
from jsonpath_ng.ext.filter import Expression as BaseExpression
from jsonpath_ng.ext.parser import ExtendedJsonPathLexer
from jsonpath_ng.ext.string import DefintionInvalid

__author__ = 'Fernando Serena'


class JsonPathLexer(ExtendedJsonPathLexer):
    """Custom WoT-lexer for JsonPath"""
    literals = ExtendedJsonPathLexer.literals
    tokens = (['BOOL'] +
              ExtendedJsonPathLexer.tokens +
              ['FILTER_OP', 'SORT_DIRECTION', 'FLOAT'])

    t_FILTER_OP = r'=(=|~)?|<=|>=|!=|<|>'

    def t_BOOL(self, t):
        r'true|false'
        t.value = True if t.value == 'true' else False
        return t

    def t_SORT_DIRECTION(self, t):
        r',?\s*(/|\\)'
        t.value = t.value[-1]
        return t

    def t_ID(self, t):
        r'@?[a-zA-Z_][a-zA-Z0-9_@\-]*'
        # NOTE(sileht): This fixes the ID expression to be
        # able to use @ for `This` like any json query
        t.type = self.reserved_words.get(t.value, 'ID')
        return t

    def t_FLOAT(self, t):
        r'-?\d+\.\d+'
        t.value = float(t.value)
        return t


class JsonPathParser(parser.ExtentedJsonPathParser):
    """Custom WOT-parser for JsonPath"""
    tokens = JsonPathLexer.tokens

    def __init__(self, debug=False, lexer_class=None):
        lexer_class = lexer_class or JsonPathLexer
        super(JsonPathParser, self).__init__(debug, lexer_class)

    def p_jsonpath_named_operator(self, p):
        "jsonpath : NAMED_OPERATOR"
        if p[1].startswith('match('):
            p[0] = Match(p[1])
        else:
            super(JsonPathParser, self).p_jsonpath_named_operator(p)

    def p_expression(self, p):
        """expression : jsonpath
                      | jsonpath FILTER_OP ID
                      | jsonpath FILTER_OP FLOAT
                      | jsonpath FILTER_OP NUMBER
                      | jsonpath FILTER_OP BOOL
        """
        if len(p) == 2:
            left, op, right = p[1], None, None
        else:
            __, left, op, right = p
        p[0] = Expression(left, op, right)


def parse(path, debug=False):
    return JsonPathParser(debug=debug).parse(path)


OPERATOR_MAP = {
    '!=': operator.ne,
    '==': operator.eq,
    '=': operator.eq,
    '<=': operator.le,
    '<': operator.lt,
    '>=': operator.ge,
    '>': operator.gt,
    '=~': operator.contains
}


class Expression(BaseExpression):
    """The JSONQuery expression for WoT"""

    def find(self, datum):
        datum = self.target.find(DatumInContext.wrap(datum))

        if not datum:
            return []
        if self.op is None:
            return datum

        found = []
        for data in datum:
            value = data.value
            if isinstance(self.value, int):
                try:
                    value = int(value)
                except ValueError:
                    continue
            elif isinstance(self.value, bool):
                try:
                    value = bool(value)
                except ValueError:
                    continue

            if OPERATOR_MAP[self.op](value, self.value):
                found.append(data)

        return found


MATCH = re.compile("match\(/(.*)/,\s+(.*)\)")


class Match(jsonpath_rw.This):
    """String matcher

    Concrete syntax is '`match(char, segment, max_split)`'
    """

    def __init__(self, method=None):
        m = MATCH.match(method)
        if m is None:
            raise DefintionInvalid("%s is not valid" % method)
        self.char = m.group(1)
        self.segment = int(m.group(2))
        self.max_split = int(m.group(3))
        self.method = method

    def find(self, datum):
        datum = jsonpath_rw.DatumInContext.wrap(datum)
        try:
            value = datum.value.split(self.char, self.max_split)[self.segment]
        except Exception:
            return []
        return [jsonpath_rw.DatumInContext.wrap(value)]

    def __eq__(self, other):
        return (isinstance(other, Match) and self.method == other.method)

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.method)

    def __str__(self):
        return '`%s`' % self.method
