# MySQL 설치 (Windows)


0. my.ini 파일 추가

1. 관리자 권한으로 CMD를 실행한 후, 초기화 명령
mysqld --initialize


2.  mysql 윈도우 서비스 등록
mysqld --install


3. 등록된 윈도우 서비스를 구동하기 위한 명령어
net start MYSQL
cf) net helpmsg % Error 발생 시

* mysql\data 폴더를 삭제한 후, 1번부터 다시 시작한다.


4. 비밀번호 찾기
data\(컴퓨터이름).err 파일 속에 비밀번호 존재


5. MYSQL 접속하기
mysql -uroot -p


6. 최초 비밀번호 변경
alter user 'root'@'localhost' IDENTIFIED WITH mysql_native_password by 'root';
flush privileges;



-----------

# MySQL Charset Setting  (한글 깨짐 현상)

MySQL을 활용하여 개발하거나 운영할 때 웹 페이지 파일에서 'UTF8'을 사용할 때, 인코딩 문제가 발생할 경우가 있습니다.

예를 들어, html, xml 등 UI 파일에서는 UTF8을 사용하는 것으로 명시하였으나 한글이 깨지는 경우입니다.


이럴 때면, DB 설정도 살펴봐야하는 부분입니다.


mysql -u'ID' -p'비밀번호'


로 mysql에 접속합니다.


이때, status를 입력하여 엔터를 치면, 서버 환경이 표현됩니다.


characterset이 UTF8이 아닌 다른 내용일 가능성이 많다.


이 부분을 해결하려면 my.ini 파일을 찾아 추가해야하는 아래와 같습니다.



[mysql]

default-character-set=utf8



[mysqld]

character-set-server=utf8

collation-server=utf8_general_ci

init_connect=SET collation_connection=utf8_general_ci

init_connect=SET NAMES utf8
