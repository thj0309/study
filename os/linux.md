##############################################################################################################################
보기
1. ubuntu 18.04 - Could not get lock /var/lib/dpkg/lock-frontend - open 오류 해결하기
2. ubuntu 18.04 - su: Authentication failure : root 계정 접속 불가시

##############################################################################################################################


본문
1. ubuntu 18.04 - Could not get lock /var/lib/dpkg/lock-frontend - open 오류 해결하기
(1) 발생원인
- /var/lib/dpkg/lock 파일이 있는 경우, 패키지와 인덱스를 업데이트 하지 않습니다
- 위의 lock 관련 파일을 삭제한 후, 업데이트하면 해결 됩니다.

(2) 해결 방법
- 프로세스 모두 종료
sudo killall apt apt-get

- "lock" 관련 파일 삭제
sudo rm -rf /var/lib/apt/lists/lock
sudo rm -rf /var/cache/apt/archives/lock
sudo rm -rf /var/lib/dpkg/lock*

- "dpkg" 재등록
sudo dpkg --configure -a

- 우분투 다시 업데이트
sudo apt update
 
 
2. ubuntu 18.04 - su: Authentication failure : root 계정 접속 불가시
(1) 원인 : root 비밀번호를 설정하지 않아서, 설정하면 됨.
(2) 해결 방법 : sudo passwd root


3. ubuntu 18.04 : 외부 아이피 확인 방법
- curl ifconfig.me
// curl 없으면, apt install curl 로 설치 진행
